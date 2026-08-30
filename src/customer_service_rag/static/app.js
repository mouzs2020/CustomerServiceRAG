/* 跨境售后知识助手 —— 原生 JavaScript，无框架、无 CDN。
   安全约定：所有动态内容一律通过 textContent 写入，
   禁止任何形式的 HTML 注入（问题、答案、reason、证据原文均为纯文本节点）。 */

(function () {
  "use strict";

  var ANSWER_TIMEOUT_MS = 180000;
  var READY_TIMEOUT_MS = 10000;
  var MAX_QUERY_LENGTH = 2000;

  var GENERIC_ERROR =
    "服务暂时不可用，请稍后重试。若持续失败，请联系维护人员并提供下方请求 ID。";
  var NETWORK_ERROR =
    "网络请求失败或超时，请检查本地服务是否在运行后重试。";
  var INVALID_INPUT_ERROR =
    "输入无效：请检查问题内容（1–2000 字符）与平台选择后重试。";

  var BLOCKED_MESSAGES = {
    blocked_invalid_entry_platform: "未识别到有效的进入平台，请选择 AliExpress 或 Temu 后重新提问。",
    blocked_missing_platform: "请先选择进入平台（AliExpress 或 Temu）后再提问。",
    blocked_multiple_platforms: "问题中涉及多个平台，请明确单一平台后重新提问。",
    blocked_platform_conflict: "问题内容与所选平台不一致，请确认平台后重新提问。",
    blocked_unrelated_question: "该问题不在跨境退款与售后规则范围内，请调整问题后再试。",
    blocked_intent_uncertain: "暂时无法判断问题意图，请补充平台与退款相关细节后重试。",
    blocked_intent_classifier_error: "意图识别服务暂时异常，请稍后重试。",
    blocked_no_matching_source: "知识库中未找到与该问题匹配的平台规则，请换一种问法。",
    blocked_invalid_evidence: "检索到的证据未通过质量检查，无法生成有依据的回答。",
    blocked_platform_evidence_mismatch: "证据与所选平台不匹配，请确认平台后重试。",
    blocked_low_relevance: "问题与平台规则的相关度过低，请补充更多细节。",
    blocked_evidence_gate_config_error: "证据检查配置异常，请联系维护人员。"
  };

  function $(id) { return document.getElementById(id); }

  var els = {
    form: $("ask-form"),
    question: $("question-input"),
    send: $("send-button"),
    charCount: $("char-count"),
    inputHint: $("input-hint"),
    status: $("service-status"),
    statusText: $("service-status-text"),
    resultPanel: $("result-panel"),
    placeholderPanel: $("placeholder-panel"),
    loading: $("loading-indicator"),
    answerBlock: $("answer-block"),
    answerText: $("answer-text"),
    citationsBlock: $("citations-block"),
    usedCitations: $("used-citations"),
    blockedBlock: $("blocked-block"),
    blockedMessage: $("blocked-message"),
    blockedStatus: $("blocked-status"),
    blockedReason: $("blocked-reason"),
    errorBlock: $("error-block"),
    errorMessage: $("error-message"),
    errorRequestId: $("error-request-id"),
    evidenceBlock: $("evidence-block"),
    evidenceList: $("evidence-list")
  };

  /* 两个进入平台 radio：loading 期间统一禁用/恢复。 */
  els.platformRadios = Array.prototype.slice.call(
    document.querySelectorAll('input[name="entry-platform"]')
  );

  var loading = false;
  /* 服务就绪状态：仅由 /ready 的结果驱动；false 时发送按钮恒禁用。 */
  var serviceReady = false;

  /* ---------- 通用渲染工具 ---------- */

  function setText(el, value) {
    // 仅 textContent：null / undefined 显示为 “-”，绝不渲染成字符串 "null"。
    el.textContent = (value === null || value === undefined || value === "") ? "-" : String(value);
  }

  function hideAll() {
    [els.loading, els.answerBlock, els.blockedBlock, els.errorBlock, els.evidenceBlock]
      .forEach(function (el) { el.hidden = true; });
  }

  function showPanel() {
    els.placeholderPanel.hidden = true;
    els.resultPanel.hidden = false;
  }

  function showPlaceholder() {
    els.resultPanel.hidden = true;
    els.placeholderPanel.hidden = false;
  }

  function showLoading() {
    hideAll();
    showPanel();
    els.loading.hidden = false;
  }

  function setLoadingState(next) {
    loading = next;
    els.question.disabled = next;
    els.platformRadios.forEach(function (radio) {
      radio.disabled = next;
    });
    els.send.disabled = !isSendEnabled();
  }

  function isInputValid() {
    return selectedPlatform() !== null && els.question.value.trim().length > 0;
  }

  function selectedPlatform() {
    var checked = els.form.querySelector('input[name="entry-platform"]:checked');
    return checked ? checked.value : null;
  }

  /* 发送按钮统一判断：非 loading + 服务就绪 + 输入合法，三者缺一不可。 */
  function isSendEnabled() {
    return !loading && serviceReady && isInputValid();
  }

  function refreshSendState() {
    if (els.inputHint.hidden === false) {
      els.inputHint.hidden = true;
    }
    els.send.disabled = !isSendEnabled();
  }

  /* ---------- fetch 封装：统一超时 ---------- */

  function fetchWithTimeout(url, options, timeoutMs) {
    var controller = new AbortController();
    var timer = setTimeout(function () { controller.abort(); }, timeoutMs);
    var init = Object.assign({}, options, { signal: controller.signal });
    return fetch(url, init).finally(function () { clearTimeout(timer); });
  }

  function requestIdOf(response) {
    return response.headers.get("X-Request-ID") || "-";
  }

  /* ---------- 服务就绪状态 ---------- */

  function renderServiceStatus(status) {
    // serviceReady 只认 ready；not_ready / checking / unknown 一律 false。
    serviceReady = status === "ready";
    els.status.classList.remove("status-loading", "status-ready", "status-not-ready");
    if (status === "ready") {
      els.status.classList.add("status-ready");
      setText(els.statusText, "服务就绪");
    } else if (status === "not_ready") {
      els.status.classList.add("status-not-ready");
      setText(els.statusText, "服务未就绪");
    } else if (status === "checking") {
      els.status.classList.add("status-loading");
      setText(els.statusText, "检测服务状态中…");
    } else {
      els.status.classList.add("status-loading");
      setText(els.statusText, "状态检测失败");
    }
    // 就绪状态变化立即影响发送按钮可用性。
    refreshSendState();
  }

  function checkServiceStatus() {
    renderServiceStatus("checking");
    fetchWithTimeout("/ready", { method: "GET" }, READY_TIMEOUT_MS)
      .then(function (response) {
        // /ready 就绪返回 200，未就绪返回 503，两者 body 均含 status 字段。
        return response.json().then(function (body) {
          var status = body && body.status === "ready" ? "ready" : "not_ready";
          renderServiceStatus(status);
        });
      })
      .catch(function () { renderServiceStatus("unknown"); });
  }

  /* ---------- 证据 / 引用渲染 ---------- */

  function renderEvidence(evidence) {
    els.evidenceList.textContent = "";
    (evidence || []).forEach(function (item) {
      var li = document.createElement("li");
      li.className = "evidence-item";

      var head = document.createElement("div");
      head.className = "evidence-head";

      var id = document.createElement("span");
      id.className = "evidence-id";
      setText(id, item.citation_id);
      head.appendChild(id);

      var platform = document.createElement("span");
      platform.className = "evidence-platform";
      setText(platform, item.platform);
      head.appendChild(platform);

      var headings = document.createElement("span");
      headings.className = "evidence-headings";
      var headingList = Array.isArray(item.headings) ? item.headings : [];
      setText(headings, headingList.length ? headingList.join(" › ") : "-");
      head.appendChild(headings);

      var quote = document.createElement("blockquote");
      quote.className = "evidence-quote";
      setText(quote, item.text);

      var scores = document.createElement("div");
      scores.className = "evidence-scores";
      var retrieveScore = document.createElement("span");
      retrieveScore.textContent = "retrieve_score: " + Number(item.retrieve_score).toFixed(3);
      var rerankScore = document.createElement("span");
      rerankScore.textContent = "rerank_score: " + Number(item.rerank_score).toFixed(3);
      scores.appendChild(retrieveScore);
      scores.appendChild(rerankScore);

      li.appendChild(head);
      li.appendChild(quote);
      li.appendChild(scores);
      els.evidenceList.appendChild(li);
    });
  }

  function renderCitations(usedCitations) {
    els.usedCitations.textContent = "";
    var citations = Array.isArray(usedCitations) ? usedCitations : [];
    els.citationsBlock.hidden = citations.length === 0;
    citations.forEach(function (citationId) {
      var chip = document.createElement("span");
      chip.className = "citation-chip";
      setText(chip, citationId);
      els.usedCitations.appendChild(chip);
    });
  }

  /* ---------- 各类结果渲染 ---------- */

  function renderReady(body) {
    hideAll();
    showPanel();
    els.answerBlock.hidden = false;
    setText(els.answerText, body.answer);
    renderCitations(body.used_citations);
    var evidence = Array.isArray(body.evidence) ? body.evidence : [];
    els.evidenceBlock.hidden = evidence.length === 0;
    if (evidence.length) { renderEvidence(evidence); }
  }

  function renderBlocked(body) {
    hideAll();
    showPanel();
    els.blockedBlock.hidden = false;
    var message = (body.answer !== null && body.answer !== undefined && body.answer !== "")
      ? body.answer
      : (BLOCKED_MESSAGES[body.status] || "暂时无法回答该问题，请调整问题或平台后重试。");
    setText(els.blockedMessage, message);
    setText(els.blockedStatus, body.status);
    setText(els.blockedReason, body.reason);
  }

  function renderError(message, requestId) {
    hideAll();
    showPanel();
    els.errorBlock.hidden = false;
    setText(els.errorMessage, message);
    setText(els.errorRequestId, requestId);
  }

  /* ---------- 提交流程 ---------- */

  function submitQuestion() {
    if (loading) { return; }

    var platform = selectedPlatform();
    var query = els.question.value.trim();

    if (!platform || !query) {
      els.inputHint.textContent = !platform
        ? "请先选择平台（AliExpress 或 Temu）。"
        : "请输入问题内容。";
      els.inputHint.hidden = false;
      return;
    }

    setLoadingState(true);
    showLoading();

    fetchWithTimeout(
      "/v1/answer",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query, entry_platform: platform })
      },
      ANSWER_TIMEOUT_MS
    ).then(function (response) {
      if (response.status === 200) {
        return response.json().then(function (body) {
          if (body && body.status === "ready_for_grounding") {
            renderReady(body);
          } else {
            renderBlocked(body || {});
          }
        });
      }
      var requestId = requestIdOf(response);
      if (response.status === 422) {
        renderError(INVALID_INPUT_ERROR, requestId);
        return undefined;
      }
      return response.json()
        .catch(function () { return null; })
        .then(function (body) {
          var reason = (body && typeof body.reason === "string" && body.reason)
            ? body.reason
            : GENERIC_ERROR;
          renderError(reason, requestId);
        });
    }).catch(function () {
      // 网络失败或超时（AbortError）：统一安全话术，不带异常原文。
      renderError(NETWORK_ERROR, "-");
    }).finally(function () {
      setLoadingState(false);
    });
  }

  /* ---------- 事件绑定 ---------- */

  els.form.addEventListener("submit", function (event) {
    event.preventDefault();
    submitQuestion();
  });

  els.form.addEventListener("change", function (event) {
    if (event.target && event.target.name === "entry-platform") {
      refreshSendState();
    }
  });

  els.question.addEventListener("input", function () {
    var length = els.question.value.length;
    els.charCount.textContent = length + " / " + MAX_QUERY_LENGTH;
    refreshSendState();
  });

  refreshSendState();
  checkServiceStatus();
})();
