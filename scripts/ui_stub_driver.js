/* Драйвер скриншотов для заглушек (scripts/ui_stub_server.py и стенд листа scripts/somm_stub.py
   подмешивают его при ?shot=…).

   Проходит сценарий так же, как человек: подтверждает 18+, выбирает фото (DataTransfer в то же
   поле <input type=file>, что открывает камеру), жмёт кнопки и примеры. Страница о нём не
   знает, в бою его нет. Готовность — атрибут data-shot="ready" у <html>, ошибка —
   data-shot="error" и текст в заголовке вкладки.

   Снимки somm_… идут на стенде листа: драйвер жмёт чипы «Спросить сомелье» и уточнения под
   ответом и ждёт конца каждого хода — метки текста, чипов и каретки. Экран ожидания скана (busy)
   снимается на 2-й секунде: часы страницы останавливаются, и счётчик не набегает, пока
   безголовый Chrome ждёт конца бюджета виртуального времени.

   После сценария драйвер меряет видимый текст (у снимков листа — только текст листа: страница
   под затемнением не видна) и пишет итог атрибутами <html> — их читает
   scripts/ui_screens.py из DOM страницы:
   - data-low — текст до 14px с контрастом ниже 4,5:1 (фон — ближайший непрозрачный или все
     цвета градиента, берётся худший);
   - data-sizes — все кегли видимого текста, px;
   - data-percent — знаки «%» вне «% об.»;
   - data-notices — сколько плашек 149-ФЗ видно на экране (у страницы и у листа). */
(function () {
  "use strict";
  var shot = new URLSearchParams(location.search).get("shot") || "";

  function wait(ms) { return new Promise(function (resolve) { setTimeout(resolve, ms); }); }
  async function until(test, label, ms) {
    var t0 = Date.now();
    while (Date.now() - t0 < (ms || 10000)) {
      if (test()) return;
      await wait(40);
    }
    throw new Error("не дождался: " + label);
  }
  function q(selector) { return document.querySelector(selector); }
  function imagesDone() {
    return Array.prototype.every.call(document.images, function (img) {
      return !img.getAttribute("src") || img.complete;
    });
  }
  function scrollTo(selector) {
    var el = q(selector);
    var top = el.getBoundingClientRect().top + window.scrollY - q(".top").offsetHeight - 8;
    window.scrollTo(0, Math.max(0, top));
  }
  function ready(selector) {
    var block = q(selector);
    return block && block.getAttribute("data-state") !== "loading";
  }

  async function pick() {
    var blob = await (await fetch("/__stub/photo")).blob();
    var file = new File([blob], "bottle.jpg", { type: blob.type || "image/jpeg" });
    var transfer = new DataTransfer();
    transfer.items.add(file);
    var input = q("#camera");
    input.files = transfer.files;
    input.dispatchEvent(new Event("change", { bubbles: true }));
  }

  // Анимации листов и плиток безголовый Chrome замораживает на первом кадре: на снимке лист
  // остаётся полупрозрачным. Для снимков они выключаются, как `animations: "disabled"` у Playwright.
  // Плавная прокрутка ленты листа там же не доезжает: новый ход остался бы ниже края экрана.
  var still = document.createElement("style");
  still.textContent = "*,*::before,*::after{animation-duration:0s!important;animation-delay:0s!important;" +
    "transition:none!important;scroll-behavior:auto!important}";
  document.head.appendChild(still);

  async function card() {
    await until(function () { return ready("#dishBlock") && ready("#similar"); }, "сомелье и похожие");
    if (shot === "taste" || shot === "example_taste") scrollTo("#taste");
    if (shot === "dishes" || shot === "example_dishes") scrollTo("#kpis");
    if (shot === "similar") scrollTo("#similar");
  }

  // ---------------------------------------------------------------- лист сомелье (стенд)
  // Каждый шаг — чип, который жмёт человек: где он стоит и его текст.
  var SOMM = {
    somm_guided: [[".stand", "Помогите выбрать"]],
    somm_want: [[".stand", "Помогите выбрать"], [".somm-turn__tail", "К мясу"]],
    somm_three: [[".stand", "Помогите выбрать"], [".somm-turn__tail", "К мясу"], [".somm-turn__tail", "Помягче"]],
    somm_dish: [[".stand", "К чему подать"], [".somm-turn__tail", "А к шашлыку?"]]
  };
  function chips(scope) { return Array.prototype.slice.call(document.querySelectorAll(scope + " .somm-chip")); }
  function tap(scope, text) {
    var chip = chips(scope).filter(function (el) { return el.textContent.trim() === text; })[0];
    if (!chip) throw new Error("нет чипа «" + text + "»");
    chip.click();
  }
  // Ход кончен: статус «на связи», под последним ответом метка и чипы уточнений, каретка допечатала.
  function answered(n) {
    var turns = document.querySelectorAll(".somm-turn"), last = turns[turns.length - 1];
    return turns.length === n && !q(".somm-status--busy") && Boolean(last.querySelector(".somm-label")) &&
      Boolean(last.querySelector(".somm-turn__tail .somm-chips")) && !last.querySelector("[aria-busy]");
  }
  async function somm() {
    await until(function () { return chips(".stand").length > 0; }, "чипы «Спросить сомелье»");
    var steps = SOMM[shot];
    if (!steps) throw new Error("нет сценария листа " + shot);
    for (var i = 0; i < steps.length; i++) {
      tap(steps[i][0], steps[i][1]);
      var n = i + 1;
      await until(function () { return answered(n); }, "ответ сомелье, ход " + n, 15000);
    }
    await until(imagesDone, "картинки");
    // Подборка: лента докручивается до конца третьей плитки — на экране все три вина.
    var feed = q(".somm-feed"), tiles = feed.querySelectorAll(".somm-tile");
    if (shot === "somm_three" && tiles.length) {
      feed.scrollTop += tiles[tiles.length - 1].getBoundingClientRect().bottom - feed.getBoundingClientRect().bottom + 16;
    }
    // Стенд под листом — служебная страница (ссылки на заглушки): на снимке под затемнением
    // её нет, безголовый Chrome без видеокарты не всегда размывает её фильтром затемнения.
    q(".stand").style.visibility = "hidden";
    await wait(150);
  }

  async function run() {
    if (shot.indexOf("somm_") === 0) { await somm(); return; }
    if (shot === "age") { await wait(200); return; }
    q("#ageYes").click();
    if (shot === "home") { await until(imagesDone, "картинки"); return; }
    if (shot.indexOf("example") === 0) {
      q("#examples .ex").click();
      await until(function () {
        return document.body.getAttribute("data-screen") === "result" && q("#wineName").textContent !== "Открываю карточку…";
      }, "карточка примера");
      await card();
      await until(imagesDone, "картинки");
      await wait(150);
      return;
    }
    if (shot === "busy") {
      // Ответ скана не приходит: экран разбора стоит, как во время настоящего скана.
      var real = window.fetch;
      window.fetch = function (url) {
        if (String(url).indexOf("/v1/scan") === 0) return new Promise(function () {});
        return real.apply(this, arguments);
      };
      await pick();
      // Все этапы пройдены к 1,9 с. Дальше часы страницы стоят: безголовый Chrome снимает экран
      // в конце бюджета виртуального времени, и счётчик иначе показал бы 20 с вместо 2 с.
      await wait(2000);
      var frozen = Date.now();
      Date.now = function () { return frozen; };
      await wait(400);
      return;
    }
    await pick();
    await until(function () {
      var screen = document.body.getAttribute("data-screen");
      return screen && screen !== "busy" && screen !== "home";
    }, "экран результата");
    if (shot === "not_found") {
      await until(function () { return ready("#nfSim"); }, "подборка по этикетке");
    } else if (document.body.getAttribute("data-screen") === "result") {
      await card();
    }
    await until(imagesDone, "картинки");
    await wait(150);
  }

  // ---------------------------------------------------------------- замеры видимого текста
  function rgb(value) {
    var m = /rgba?\(([^)]+)\)/.exec(value || "");
    if (!m) return null;
    var p = m[1].split(",").map(parseFloat);
    return { c: p.slice(0, 3), a: p.length > 3 ? p[3] : 1 };
  }
  function lum(c) {
    var v = c.map(function (x) { x /= 255; return x <= 0.03928 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4); });
    return 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2];
  }
  function ratio(a, b) {
    var l1 = lum(a), l2 = lum(b);
    return (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05);
  }
  function blend(fg, bg) {
    return fg.c.map(function (x, i) { return x * fg.a + bg[i] * (1 - fg.a); });
  }
  // Фоны под элементом: ближайший непрозрачный цвет или все цвета градиента.
  function backgrounds(el) {
    for (var e = el; e && e.nodeType === 1; e = e.parentElement) {
      var cs = getComputedStyle(e);
      if (cs.backgroundImage && cs.backgroundImage.indexOf("gradient") >= 0) {
        var stops = (cs.backgroundImage.match(/rgba?\([^)]+\)/g) || []).map(rgb)
          .filter(function (s) { return s && s.a > 0.5; });
        if (stops.length) return stops.map(function (s) { return s.c; });
      }
      var bg = rgb(cs.backgroundColor);
      if (bg && bg.a > 0.5) return [bg.c];
    }
    return [[254, 253, 250]];
  }
  function visible(el) {
    if (el.closest(".hidden") || el.closest("script, style")) return false;
    return el.getClientRects().length > 0 && getComputedStyle(el).visibility !== "hidden";
  }
  function measure() {
    var low = [], sizes = {}, texts = [];
    // Лист закрывает страницу затемнением: у снимков листа меряется только он.
    var root = (shot.indexOf("somm_") === 0 && q(".somm-sheet")) || document.body;
    var walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    var node;
    while ((node = walker.nextNode())) {
      var el = node.parentElement, text = node.nodeValue.trim();
      if (!text || !el || !visible(el)) continue;
      texts.push(text);
      var cs = getComputedStyle(el);
      var size = parseFloat(cs.fontSize);
      sizes[Math.round(size)] = true;
      if (size > 14.5) continue;
      var fg = rgb(el.namespaceURI === "http://www.w3.org/2000/svg" ? cs.fill : cs.color);
      if (!fg) continue;
      var worst = Math.min.apply(null, backgrounds(el).map(function (bg) { return ratio(blend(fg, bg), bg); }));
      if (worst < 4.5) low.push(text.slice(0, 32) + " (" + worst.toFixed(2) + ")");
    }
    var all = texts.join(" ");
    var notices = Array.prototype.filter.call(document.querySelectorAll(".notice, .somm-notice"), visible).length;
    var root = document.documentElement;
    root.setAttribute("data-low", low.join(" | "));
    root.setAttribute("data-sizes", Object.keys(sizes).sort(function (a, b) { return b - a; }).join(","));
    root.setAttribute("data-percent", String((all.match(/%(?!\s?об\.)/g) || []).length));
    root.setAttribute("data-notices", String(notices));
  }

  run().then(function () {
    measure();
    document.documentElement.setAttribute("data-shot", "ready");
  }).catch(function (error) {
    document.documentElement.setAttribute("data-shot", "error");
    document.title = "driver: " + error.message;
    console.error(error);
  });
})();
