async () => {
  // Keep language changes in the browser so uploads, settings and playback survive.
  if (window.dbsSetLanguage) {
    await window.dbsSetLanguage("en");
    return;
  }

  let language = "en";
  let localeModulePromise;
  let scheduled = false;
  let warned = false;
  const knownFrames = new WeakSet();
  const textSelector = "[data-en][data-zh]";
  const ariaSelector = "[data-aria-en][data-aria-zh]";
  const nativeLanguage = () => language === "zh" ? "zh-CN" : "en";

  function studioFrames() {
    return Array.from(document.querySelectorAll("iframe.studio-frame")).filter(frame => {
      try {
        return new URL(frame.src || window.location.href, window.location.href).origin === window.location.origin;
      } catch {
        return false;
      }
    });
  }

  function sendLanguage(frame) {
    frame.contentWindow?.postMessage({type: "dbs-language", language}, window.location.origin);
  }

  function syncFrames() {
    for (const frame of studioFrames()) {
      if (!knownFrames.has(frame)) {
        knownFrames.add(frame);
        frame.addEventListener("load", () => sendLanguage(frame));
        sendLanguage(frame);
      }
    }
  }

  function translateUpload() {
    const upload = document.querySelector("#video-upload");
    if (!upload) return;
    const isChinese = language === "zh";
    const button = upload.querySelector("button.upload-container");
    const input = upload.querySelector('input[data-testid="file-upload"]');
    for (const [element, label] of [
      [button, isChinese ? "点击上传或拖放音频或视频" : "Click to upload or drop audio or video"],
      [input, isChinese ? "上传音频或视频" : "Upload audio or video"]
    ]) {
      if (element && element.getAttribute("aria-label") !== label) element.setAttribute("aria-label", label);
    }
    const text = button?.querySelector('[data-testid="upload-text"]');
    if (!text) return;
    // Gradio snapshots these strings on mount. Edit only its existing text
    // nodes, preserving the upload icon, input, button and their event handlers.
    const slots = Array.from(text.childNodes).filter(node => node.nodeType === 3 && node.nodeValue.trim());
    if (slots.length === 2) {
      const labels = isChinese ? ["将音频或视频拖放到此处 ", " 点击上传"] : ["Drop audio or video here ", " Click to upload"];
      slots.forEach((node, index) => {
        if (node.nodeValue !== labels[index]) node.nodeValue = labels[index];
      });
    }
    const separator = text.querySelector("span.or");
    const separatorText = separator && Array.from(separator.childNodes).find(node => node.nodeType === 3 && node.nodeValue.trim());
    const separatorLabel = isChinese ? "- 或 -" : "- or -";
    if (separatorText && separatorText.nodeValue !== separatorLabel) separatorText.nodeValue = separatorLabel;
  }

  function translatePage() {
    if (document.documentElement.lang !== nativeLanguage()) document.documentElement.lang = nativeLanguage();
    for (const element of document.querySelectorAll(textSelector)) {
      const translated = element.getAttribute(`data-${language}`);
      if (element.textContent !== translated) element.textContent = translated;
    }
    for (const element of document.querySelectorAll(ariaSelector)) {
      const translated = element.getAttribute(`data-aria-${language}`);
      if (element.getAttribute("aria-label") !== translated) element.setAttribute("aria-label", translated);
    }
    translateUpload();
    syncFrames();
  }

  function scheduleTranslation() {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(() => {
      scheduled = false;
      translatePage();
    });
  }

  async function syncNativeLocale() {
    try {
      localeModulePromise ??= import("__GRADIO_LOCALE_MODULE__");
      const localeModule = await localeModulePromise;
      // Read the latest selection after loading, never the request that started it.
      localeModule.changeLocale(nativeLanguage());
    } catch (error) {
      localeModulePromise = undefined;
      if (!warned) {
        warned = true;
        console.warn("The studio language changed, but Gradio's locale could not be loaded.", error);
      }
    }
  }

  window.addEventListener("message", event => {
    if (event.origin !== window.location.origin || event.data?.type !== "dbs-language-ready") return;
    const frame = studioFrames().find(candidate => candidate.contentWindow === event.source);
    if (frame) sendLanguage(frame);
  });

  const observer = new MutationObserver(records => {
    // Gradio loads locale bundles asynchronously. A late older request must not
    // overwrite a newer selection; its locale store also updates the html lang.
    if (records.some(record => record.target === document.documentElement && record.attributeName === "lang")
        && document.documentElement.lang !== nativeLanguage()) {
      void syncNativeLocale();
    }
    scheduleTranslation();
  });
  observer.observe(document.documentElement, {
    subtree: true,
    childList: true,
    characterData: true,
    attributes: true,
    attributeFilter: ["data-en", "data-zh", "data-aria-en", "data-aria-zh", "lang"]
  });

  window.dbsSetLanguage = async requestedLanguage => {
    language = requestedLanguage === "zh" ? "zh" : "en";
    window.dbsLanguage = language;
    translatePage();
    for (const frame of studioFrames()) sendLanguage(frame);
    await syncNativeLocale();
    return language;
  };

  await window.dbsSetLanguage("en");
}
