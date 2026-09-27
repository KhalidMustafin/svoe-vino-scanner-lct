/* «Сомелье» сканера: заметка, профиль вкуса и лист-разговор (docs/api-sommelier.md, §8).

   Файл без сборки и без внешних адресов. Страница index.html подключает его рядом с somm.css;
   при загрузке он только объявляет window.SommUI и в DOM не лезет. Всё, что пришло из API,
   попадает в страницу текстом (textContent): разметки строкой здесь нет вовсе, значки и «роза
   ветров» собираются узлами SVG.

   Лист повторяет порядок ответа «Лозы» (Answer.tsx): пузырь вопроса → настоящие этапы сервера →
   «Коротко» с основанием → блюда и вина с чипами правил → сравнение с исходным вином → уточнения
   только под последним ответом → метка текста. Этапы приходят из потока и не выдумываются: этап
   виден ровно столько, сколько идёт на сервере, строк-утешений нет.

   Шаблон ответа виден сразу по событию facts с меткой «Текст и подбор — алгоритм». Живой текст
   модели приходит одним событием text уже проверенным; каретка только печатает его на экран.
   Разговор живёт в памяти вкладки и никуда не пишется: ни в хранилище браузера, ни в консоль. */
(function () {
  "use strict";

  // ---------------------------------------------------------------- тексты и константы
  var LABEL_AI = "Текст — ИИ, подбор — алгоритм";
  var LABEL_ALGO = "Текст и подбор — алгоритм";
  var ASK = "/v1/sommelier/ask?order=";
  var WINES = "/v1/wines/";
  var MAX_QUESTION = 200;
  var CARET_MS = 18;
  // Порядок осей — app/recommend/profile.py (AXES); на «розе ветров» — от 12 часов по часовой.
  var AXES = ["sweetness", "acidity", "tannin", "body", "alcohol", "oak", "aroma_intensity", "effervescence"];
  var SCALE_AXES = ["sweetness", "alcohol", "acidity", "tannin", "body"];
  // Источник оси: факт выгрузки — сплошная точка; оценка по сорту или по стилю — полая.
  var SOURCE_TEXT = { catalog: "из карточки", grape: "по сорту", style: "по стилю" };
  var RULE_SOURCE = { grape: "по сорту", type: "типично для стиля", style: "типично для стиля" };
  var VERDICT = { yes: "подходит", caveat: "подходит с оговоркой", no: "скорее нет", neutral: "правила молчат" };
  var LEGEND_CATALOG = "из карточки каталога";
  var LEGEND_GRAPE = "по сорту — типично для сорта, не замер этого вина";
  var LEGEND_STYLE = "по стилю — типично для стиля, не замер этого вина";
  var NOTICE = "Применяются рекомендательные технологии";
  var PLAIN_NOTE = "Обычная сортировка: по названию, без подбора";
  // Направления — коротким рядом под заметкой карточки: «помягче» одним касанием, без поиска.
  var DIRECTIONS = ["softer", "fresher", "sweeter"];
  var HARM = "Чрезмерное употребление алкоголя вредит вашему здоровью. 18+";
  var HELLO = "Спросите об этом вине: к чему подать, как подать и чем заменить.";
  var FAIL_TEXT = "Не получилось собрать ответ — повторите вопрос";
  var LOST_TEXT = "Связь прервалась — повторите вопрос";
  var CUT_TEXT = "Ответ прерван — задайте вопрос снова";
  var GONE_TEXT = "Этого вина нет в каталоге сомелье — откройте карточку заново";
  var INPUT_OFF = "вопрос текстом выключен";
  var INPUT_OFF_TEXT = "Вопрос текстом выключен — выберите вопрос ниже";
  var DEFAULT_CHIPS = [
    { id: "what_to_eat", text: "К чему подать" },
    { id: "serve", text: "Как подать" },
    { id: "replace", text: "Чем заменить" }
  ];
  var CHIP_TEXT = {
    what_to_eat: "К чему подать", serve: "Как подать", softer: "Помягче", fresher: "Посвежее",
    replace: "Чем заменить", guided: "Помогите выбрать"
  };
  // «Роза ветров»: холст, центр, радиус и пол радиуса — как у ProfileRadar «Лозы» (договор §8.1).
  var RW = 260, RH = 220, RCX = 130, RCY = 110, RR = 66, RMIN = 0.14;
  var ICONS = {
    camera: ["M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z",
      "M9 13a3 3 0 1 0 6 0a3 3 0 1 0-6 0"],
    close: ["M18 6 6 18", "m6 6 12 12"],
    up: ["m5 12 7-7 7 7", "M12 19V5"],
    check: ["M20 6 9 17l-5-5"],
    info: ["M2 12a10 10 0 1 0 20 0a10 10 0 1 0-20 0", "M12 16v-4", "M12 8h.01"],
    spark: ["M9.94 15.5a2 2 0 0 0-1.44-1.44l-6.13-1.58a.5.5 0 0 1 0-.96L8.5 9.94A2 2 0 0 0 9.94 8.5l1.58-6.13a.5.5 0 0 1 .96 0l1.58 6.13a2 2 0 0 0 1.44 1.44l6.13 1.58a.5.5 0 0 1 0 .96L15.5 14.06a2 2 0 0 0-1.44 1.44l-1.58 6.13a.5.5 0 0 1-.96 0z",
      "M20 3v4", "M22 5h-4"]
  };
  // Силуэт бутылки вместо фото, которого нет: стекло, колпачок, пунктир этикетки и блик — как
  // бутылка в рамке скана на главной, а не значок битой картинки.
  var BOTTLE = [
    ["path", "somm-bottle__glass", { d: "M15.5 3.5h9v19c0 3.4 1.9 5.9 4 8.6 2.1 2.7 3.5 5.6 3.5 9.4v60a5 5 0 0 1-5 5h-14a5 5 0 0 1-5-5V40.5c0-3.8 1.4-6.7 3.5-9.4 2.1-2.7 4-5.2 4-8.6z" }],
    ["rect", "somm-bottle__cap", { x: 15.5, y: 3.5, width: 9, height: 11, rx: 1.5 }],
    ["rect", "somm-bottle__label", { x: 13, y: 58, width: 14, height: 26, rx: 2.5 }],
    ["path", "somm-bottle__shine", { d: "M10.5 45v47" }]
  ];

  // ---------------------------------------------------------------- хозяин и состояние
  // Колбэки страницы (договор §8.1); до init — безопасные заглушки.
  var host = {
    order: function () { return "reco"; },
    setOrder: function () {},
    openWine: function () {},
    scan: function () {},
    onSheet: function () {},
    localUrl: function (url) {
      return typeof url === "string" && /^\/v1\/[A-Za-z0-9._\/-]+$/.test(url) && url.indexOf("..") < 0 ? url : null;
    },
    portalUrl: function () { return null; }
  };
  var sheet = null;       // открытый (или спрятанный) лист одного вина
  var svgNs = null;
  var timers = [];        // каретки: обрываются вместе с листом
  var noted = null;       // ряд направлений под заметкой {slug, ids}: во «Спросить сомелье» их нет

  // ---------------------------------------------------------------- мелочи
  function str(value) { return typeof value === "string" ? value : ""; }
  function list(value) { return Array.isArray(value) ? value : []; }
  function isObj(value) { return value !== null && typeof value === "object" && !Array.isArray(value); }
  function clamp(value) { return Math.min(Math.max(value, 0), 1); }
  function round(value) { return Math.round(value * 10) / 10; }
  // «16–18 °C» не рвётся в узкой ленте, тире не начинает строку (как на странице): у диапазона —
  // связки без ширины, перед «°C» и «—» — неразрывный пробел.
  function nb(text) { return str(text).replace(/(\d)–(\d)/g, "$1\u2060–\u2060$2").replace(/ (°C|—)/g, "\u00a0$1"); }
  // Значение из своей таблицы: ключ из API — только среди своих ключей («constructor» — не код).
  function own(map, key) { return typeof key === "string" && Object.prototype.hasOwnProperty.call(map, key) ? map[key] : ""; }

  function call(name, arg) {
    try { return host[name](arg); } catch (e) { return null; }
  }
  function order() { return call("order") === "plain" ? "plain" : "reco"; }
  function reducedMotion() {
    try { return window.matchMedia("(prefers-reduced-motion: reduce)").matches; } catch (e) { return false; }
  }

  // Узел с классом и текстом; текст — только textContent.
  function h(tag, cls, text) {
    var el = document.createElement(tag);
    if (cls) el.className = cls;
    if (text !== undefined && text !== null) el.textContent = String(text);
    return el;
  }
  function add(parent) {
    for (var i = 1; i < arguments.length; i++) {
      var kid = arguments[i];
      if (kid) parent.appendChild(typeof kid === "string" ? document.createTextNode(kid) : kid);
    }
    return parent;
  }
  function attrs(el, map) {
    Object.keys(map).forEach(function (key) { el.setAttribute(key, String(map[key])); });
    return el;
  }
  function clear(el) {
    while (el.firstChild) el.removeChild(el.firstChild);
    return el;
  }
  function detach(el) {
    if (el && el.parentNode) el.parentNode.removeChild(el);
  }
  function button(cls, label) {
    var el = h("button", cls);
    el.type = "button";
    if (label) el.setAttribute("aria-label", label);
    return el;
  }

  function svg(tag, map) {
    if (!svgNs) {
      // Пространство имён SVG берём у узла, который разобрал сам браузер: адреса в файле нет.
      svgNs = new DOMParser().parseFromString("<svg></svg>", "text/html").body.firstChild.namespaceURI;
    }
    return attrs(document.createElementNS(svgNs, tag), map || {});
  }
  function icon(name) {
    var root = svg("svg", { viewBox: "0 0 24 24", "class": "somm-ico", "aria-hidden": "true", focusable: "false" });
    (ICONS[name] || []).forEach(function (d) { root.appendChild(svg("path", { d: d })); });
    return root;
  }
  function bottle() {
    var root = svg("svg", { viewBox: "0 0 40 110", "class": "somm-bottle", "aria-hidden": "true", focusable: "false" });
    BOTTLE.forEach(function (part) {
      var map = { "class": part[1] };
      Object.keys(part[2]).forEach(function (key) { map[key] = part[2][key]; });
      root.appendChild(svg(part[0], map));
    });
    return root;
  }
  // Фото вина: только наш путь /v1/…; нет или не загрузилось — силуэт бутылки.
  function photo(url, cls) {
    var box = h("span", cls);
    var src = call("localUrl", url);
    if (typeof src !== "string" || !src) return add(box, bottle());
    var img = h("img");
    img.alt = "";
    img.decoding = "async";
    img.loading = "lazy";
    img.addEventListener("error", function () { detach(img); box.appendChild(bottle()); });
    img.src = src;
    return add(box, img);
  }
  // Подложка фото — токен оформления по цвету вина, а не поле ответа (договор §8.2).
  function tint(style) {
    var low = str(style).toLowerCase();
    if (low.indexOf("красн") === 0) return "red";
    if (low.indexOf("бел") === 0) return "white";
    if (low.indexOf("роз") === 0) return "rose";
    if (low.indexOf("оранж") === 0) return "orange";
    return "none";
  }
  // Положение точки на шкале — переменной CSS: знак процента живёт только в somm.css.
  function dot(value, cls) {
    var el = h("span", cls);
    el.style.setProperty("--at", String(round(clamp(value) * 100) / 100));
    return el;
  }

  // ---------------------------------------------------------------- профиль
  // Ось профиля или null: значение без источника (и наоборот) не рисуется нигде.
  function axisOf(profile, axis) {
    var rows = list(profile);
    for (var i = 0; i < rows.length; i++) {
      var row = rows[i];
      if (!isObj(row) || row.axis !== axis) continue;
      if (!(typeof row.value === "number" && isFinite(row.value) && own(SOURCE_TEXT, row.source))) return null;
      return { axis: axis, label: str(row.label), left: str(row.left), right: str(row.right),
        value: clamp(row.value), source: row.source };
    }
    return null;
  }
  // Словами полюсов, без чисел: для aria-label шкал и «розы ветров».
  function pole(a) {
    return a.value < 0.4 ? a.left : a.value > 0.6 ? a.right : "между «" + a.left + "» и «" + a.right + "»";
  }
  function words(a) {
    return a.label.toLowerCase() + " — " + pole(a) + (a.source !== "catalog" ? ", " + SOURCE_TEXT[a.source] : "");
  }
  // Модификатор класса точки: у факта выгрузки его нет, у оценки — источник (полая точка).
  function est(a) { return a.source !== "catalog" ? a.source : ""; }
  // Отрезок «розы» к вершине-оценке — пунктир; имя — по более слабой вершине (сорт < стиль).
  function edgeOf(a, b) {
    return a.source === "style" || b.source === "style" ? "style" : a.source === "grape" || b.source === "grape" ? "grape" : "";
  }

  function point(index, value) {
    var angle = Math.PI * 2 * index / AXES.length - Math.PI / 2;
    var r = RR * Math.max(value, RMIN);
    return [round(RCX + r * Math.cos(angle)), round(RCY + r * Math.sin(angle))];
  }
  // Контур одного вина: вершина-оценка полая, отрезки к ней — пунктир.
  function shape(root, slots, key, anchor) {
    var pts = slots.map(function (slot) { return point(slot.index, slot[key].value); });
    if (!anchor) {
      root.appendChild(svg("polygon", { points: pts.map(function (p) { return p.join(","); }).join(" "),
        "class": "somm-radar__area" }));
    }
    slots.forEach(function (slot, i) {
      var j = i + 1 < slots.length ? i + 1 : 0;   // последняя вершина замыкается на первую
      var edge = edgeOf(slot[key], slots[j][key]);
      root.appendChild(svg("line", { x1: pts[i][0], y1: pts[i][1], x2: pts[j][0], y2: pts[j][1],
        "class": "somm-radar__edge" + (anchor ? " somm-radar__edge--orig" : "") + (edge ? " somm-radar__edge--" + edge : "") }));
    });
    if (anchor) return;
    slots.forEach(function (slot, i) {
      root.appendChild(svg("circle", { cx: pts[i][0], cy: pts[i][1], r: 3,
        "class": "somm-radar__dot" + (est(slot[key]) ? " somm-radar__dot--" + est(slot[key]) : "") }));
    });
  }

  // Вершины «розы ветров»: оси, известные у вина (и у якоря, если он есть), в порядке AXES.
  function slotsOf(profile, anchor) {
    var slots = [];
    AXES.forEach(function (axis, index) {
      var own = axisOf(profile, axis);
      var other = anchor ? axisOf(anchor, axis) : null;
      if (own && (!anchor || other)) slots.push({ index: index, own: own, other: other });
    });
    return slots;
  }
  // Какие метки нарисованы: точки по источникам и отрезки (edge-: сплошной — обе вершины из
  // карточки, пунктир — к вершине «по сорту» или «по стилю»). Легенда называет только их.
  function kindsOf(slots) {
    var kinds = {};
    slots.forEach(function (slot, i) {
      kinds[slot.own.source] = true;
      kinds["edge-" + (edgeOf(slot.own, slots[i + 1 < slots.length ? i + 1 : 0].own) || "solid")] = true;
    });
    return kinds;
  }

  // «Роза ветров» (P0 карточки): ось без значения не рисуется вовсе, меньше трёх осей — false.
  // compare = {profile, name} — исходное вино (якорь): серая полоса под контуром предложенного
  // видна, даже когда профили совпадают.
  function radar(el, profile, compare) {
    if (!el) return false;
    clear(el);
    var anchor = isObj(compare) && Array.isArray(compare.profile) ? compare.profile : null;
    var slots = slotsOf(profile, anchor);
    if (slots.length < 3) {
      el.classList.remove("somm-radar");
      return false;
    }
    el.classList.add("somm-radar");
    var said = slots.map(function (slot) { return words(slot.own); }).join("; ");
    if (anchor) said += "; серым контуром для сравнения — " + (str(compare.name) || "исходное вино");
    var root = svg("svg", { viewBox: "0 0 " + RW + " " + RH, "class": "somm-radar__svg", role: "img",
      "aria-label": "Профиль вкуса: " + said });
    [1 / 3, 2 / 3, 1].forEach(function (ring) {
      root.appendChild(svg("circle", { cx: RCX, cy: RCY, r: round(RR * ring), "class": "somm-radar__ring" }));
    });
    slots.forEach(function (slot) {
      var end = point(slot.index, 1);
      root.appendChild(svg("line", { x1: RCX, y1: RCY, x2: end[0], y2: end[1], "class": "somm-radar__ray" }));
      var angle = Math.PI * 2 * slot.index / AXES.length - Math.PI / 2;
      var cos = Math.cos(angle), sin = Math.sin(angle);
      var label = svg("text", { x: round(RCX + (RR + 12) * cos), y: round(RCY + (RR + 12) * sin + 4 + 6 * sin),
        "text-anchor": Math.abs(cos) < 0.2 ? "middle" : cos > 0 ? "start" : "end", "class": "somm-radar__label" });
      label.textContent = slot.own.label;
      root.appendChild(label);
    });
    if (anchor) shape(root, slots, "other", true);
    shape(root, slots, "own", false);
    el.appendChild(root);
    return true;
  }

  // Пункт легенды: один или несколько значков (точка, кольцо, отрезок) и подпись.
  function mark(kinds, text) {
    var box = h("span", "somm-legend__item");
    [].concat(kinds).forEach(function (kind) {
      box.appendChild(attrs(h("i", "somm-legend__mark somm-legend__mark--" + kind), { "aria-hidden": "true" }));
    });
    return add(box, text);
  }
  // Пункты легенды точек — только нарисованные виды; у «розы» — вместе с их отрезками.
  function sourceMarks(kinds) {
    return [["catalog", "solid", LEGEND_CATALOG], ["grape", "dash", LEGEND_GRAPE], ["style", "dash", LEGEND_STYLE]]
      .filter(function (row) { return kinds[row[0]]; }).map(function (row) {
        var line = kinds["edge-" + (row[0] === "catalog" ? "solid" : row[0])];
        return mark(line ? [row[0], row[1]] : row[0], row[2]);
      });
  }
  function legend(compareName, kinds) {
    var box = h("p", "somm-legend");
    if (compareName) box.appendChild(mark("orig", compareName));
    return add.apply(null, [box].concat(sourceMarks(kinds)));
  }
  function track(own, other, cls) {
    var line = attrs(h("span", "somm-track" + (cls ? " " + cls : "")), { role: "img",
      "aria-label": words(own) + (other ? "; у исходного — " + pole(other) : "") });
    if (other) line.appendChild(dot(other.value, "somm-point somm-point--orig"));
    line.appendChild(dot(own.value, "somm-point" + (est(own) ? " somm-point--" + est(own) : "")));
    return line;
  }

  // Двухполюсные шкалы. opts: {axes?, compare?: {profile, name}, legend?: false}.
  function scales(el, profile, opts) {
    if (!el) return false;
    opts = isObj(opts) ? opts : {};
    clear(el);
    var compare = isObj(opts.compare) && Array.isArray(opts.compare.profile) ? opts.compare : null;
    var fizz = axisOf(profile, "effervescence");
    var axes = Array.isArray(opts.axes) ? opts.axes
      : fizz && fizz.value > 0 ? SCALE_AXES.concat("effervescence") : SCALE_AXES;
    var rows = 0, kinds = {};
    axes.forEach(function (axis) {
      var own = axisOf(profile, axis);
      var other = compare ? axisOf(compare.profile, axis) : null;
      if (!own || (compare && !other)) return;
      rows += 1;
      kinds[own.source] = true;
      el.appendChild(add(h("div", "somm-scale"),
        add(h("div", "somm-scale__head"), h("span", "somm-scale__name", own.label),
          h("span", "somm-scale__src", SOURCE_TEXT[own.source])),
        add(h("div", "somm-scale__line"), attrs(h("span", "somm-scale__pole", own.left), { "aria-hidden": "true" }),
          track(own, other),
          attrs(h("span", "somm-scale__pole somm-scale__pole--r", own.right), { "aria-hidden": "true" }))));
    });
    el.classList.toggle("somm-scales", rows > 0);
    if (rows && opts.legend !== false) el.appendChild(legend(compare ? str(compare.name) : "", kinds));
    return rows > 0;
  }

  // ---------------------------------------------------------------- блоки карточки
  function labelLine(ai) {
    var line = h("p", "somm-label" + (ai ? " somm-label--ai" : ""));
    if (ai) line.appendChild(icon("spark"));
    return add(line, ai ? LABEL_AI : LABEL_ALGO);
  }

  // «Сомелье · коротко» — всегда шаблон из фактов с меткой алгоритма.
  function note(el, data) {
    if (!el) return false;
    clear(el);
    noted = null;
    var n = isObj(data) && isObj(data.note) ? data.note : null;
    if (!n || !str(n.text)) {
      el.hidden = true;
      return false;
    }
    el.hidden = false;
    el.classList.add("somm-note");
    var head = add(h("div", "somm-note__head"), attrs(h("span", "somm-dot"), { "aria-hidden": "true" }),
      h("span", "somm-eyebrow", "Сомелье · коротко"));
    if (str(n.basis)) head.appendChild(h("span", "somm-note__basis", n.basis));
    add(el, head, h("p", "somm-verdict", nb(n.text)), labelLine(n.generated === true && n.label === LABEL_AI));
    // Короткий ряд направлений из чипов ответа: «помягче» — одним касанием, лист с этим чипом.
    var ways = list(data.chips).filter(function (chip) { return isObj(chip) && DIRECTIONS.indexOf(chip.id) >= 0; });
    var row = str(data.slug) ? chipRow(ways, function (chip) { open(data.slug, openOpts(data, { chip: chip })); }) : null;
    if (row) {
      el.appendChild(attrs(row, { "class": "somm-chips somm-note__chips" }));
      noted = { slug: data.slug, ids: ways.map(function (chip) { return chip.id; }) };
    }
    return true;
  }

  // Реплика персоны: точка-аватар и текст (главная, check, not_found).
  function say(el, text) {
    if (!el) return;
    clear(el);
    el.classList.add("somm-say");
    add(el, attrs(h("span", "somm-say__dot"), { "aria-hidden": "true" }),
      add(h("div", "somm-say__body"), h("span", "somm-eyebrow", "Сомелье"), h("p", "somm-say__text", str(text))));
  }

  function chipRow(chips, onPick) {
    var row = h("div", "somm-chips");
    chips.forEach(function (chip) {
      if (!isObj(chip) || !str(chip.id) || !str(chip.text)) return;
      var el = add(button("somm-chip"), chip.text);
      el.addEventListener("click", function () { onPick(chip); });
      row.appendChild(el);
    });
    return row.firstChild ? row : null;
  }

  // Вопрос: обрезка краёв, схлопывание пробелов, без управляющих символов, до 200 знаков.
  function clean(text) {
    return String(text || "").replace(/[\u0000-\u001f\u007f]/g, " ").replace(/\s+/g, " ").trim().slice(0, MAX_QUESTION);
  }
  function askForm(onSend, onCamera) {
    var form = h("form", "somm-ask");
    form.setAttribute("novalidate", "");
    if (onCamera) {
      var cam = add(button("somm-round somm-round--soft", "Сфотографировать другую бутылку"), icon("camera"));
      cam.addEventListener("click", onCamera);
      form.appendChild(cam);
    }
    var input = attrs(h("input", "somm-ask__input"), { type: "text", maxlength: MAX_QUESTION, autocomplete: "off",
      enterkeyhint: "send", placeholder: "Спросить сомелье…", "aria-label": "Вопрос сомелье" });
    var send = add(button("somm-round somm-round--accent", "Отправить"), icon("up"));
    send.type = "submit";
    add(form, input, send);
    form.addEventListener("submit", function (event) {
      event.preventDefault();
      var text = clean(input.value);
      if (!text) return;
      input.value = "";
      onSend(text);
    });
    return form;
  }

  function openOpts(data, extra) {
    var opts = { name: data.name, winery: data.winery, photo_url: data.photo_url, data: data };
    Object.keys(extra).forEach(function (key) { opts[key] = extra[key]; });
    return opts;
  }
  // «Спросить сомелье» на карточке: чипы входа и поле (если data.input). Направления, уже
  // стоящие рядом под заметкой того же вина, второй раз не повторяются.
  function entry(el, data) {
    if (!el) return false;
    clear(el);
    if (!isObj(data) || !str(data.slug)) {
      el.hidden = true;
      return false;
    }
    el.hidden = false;
    el.classList.add("somm-entry");
    var skip = noted && noted.slug === data.slug ? noted.ids : [];
    var chips = list(data.chips).filter(function (chip) { return !isObj(chip) || skip.indexOf(chip.id) < 0; });
    add(el, h("p", "somm-eyebrow", "Спросить сомелье"),
      chipRow(chips, function (chip) { open(data.slug, openOpts(data, { chip: chip })); }));
    if (data.input === true) {
      el.appendChild(askForm(function (text) { open(data.slug, openOpts(data, { question: text })); }, null));
    }
    return true;
  }

  // ---------------------------------------------------------------- лист
  function build(slug, opts) {
    var name = str(opts.name) || "это вино";
    var s = { slug: slug, name: name, winery: str(opts.winery), data: null, context: null, turn: null, turns: 0, ctrl: null,
      notice: false, shown: false, winesReq: null, intro: null };
    s.wrap = attrs(h("div", "somm-wrap"), { role: "dialog", "aria-modal": "true", "aria-labelledby": "somm-title" });
    var shade = h("div", "somm-shade");
    shade.addEventListener("click", function () { close(); });
    s.panel = attrs(h("section", "somm-sheet"), { tabindex: "-1" });
    s.status = h("span", "somm-status");
    s.statusText = document.createTextNode("на связи");
    add(s.status, attrs(h("span", "somm-status__dot"), { "aria-hidden": "true" }), s.statusText);
    var shut = add(button("somm-round somm-round--ghost", "Закрыть"), icon("close"));
    shut.addEventListener("click", function () { close(); });
    var who = add(h("div", "somm-who"), attrs(h("b", "somm-who__title", "Сомелье"), { id: "somm-title" }),
      h("span", "somm-who__wine", "о " + name));
    var head = add(h("header", "somm-head"), photo(opts.photo_url, "somm-thumb"), who, s.status, shut);
    s.feed = attrs(h("div", "somm-feed"), { role: "log", "aria-live": "polite" });
    s.harm = h("p", "somm-harm", HARM);
    s.feed.appendChild(s.harm);
    s.foot = h("div", "somm-foot");
    add(s.panel, attrs(h("div", "somm-grip"), { "aria-hidden": "true" }), head, s.feed, s.foot);
    add(s.wrap, shade, s.panel);
    s.onKey = function (event) { if (event.key === "Escape") close(); else if (event.key === "Tab") keepFocus(event); };
    sheet = s;
    setInput(false);
  }
  // Лист модальный: Tab с краёв листа ходит по кругу, а не на страницу под затемнением.
  function keepFocus(event) {
    var items = Array.prototype.filter.call(sheet.panel.querySelectorAll("button, input"),
      function (el) { return !el.disabled && el.getClientRects().length > 0; });
    if (!items.length) return;
    var now = document.activeElement, inside = sheet.panel.contains(now) && now !== sheet.panel;
    if (inside && now !== (event.shiftKey ? items[0] : items[items.length - 1])) return;
    event.preventDefault();
    (event.shiftKey ? items[items.length - 1] : items[0]).focus();
  }

  // Поле внизу: камера всегда, ввод — только при input: true из GET …/sommelier.
  function setInput(on) {
    clear(sheet.foot);
    if (on) {
      sheet.foot.appendChild(askForm(function (text) { ask({ question: text, text: text }); }, toScan));
    } else {
      var cam = add(button("somm-camera"), icon("camera"), "Сфотографировать другую бутылку");
      cam.addEventListener("click", toScan);
      sheet.foot.appendChild(cam);
    }
  }
  function toScan() {
    hide();
    call("scan");
  }

  function show() {
    if (sheet.shown) return;
    sheet.shown = true;
    sheet.back = document.activeElement;
    sheet.overflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    if (sheet.paint) sheet.paint();   // сортировку могли сменить на карточке: переключатель — в тон
    document.body.appendChild(sheet.wrap);
    document.addEventListener("keydown", sheet.onKey);
    try { sheet.panel.focus({ preventScroll: true }); } catch (e) { /* старый браузер */ }
    call("onSheet", true);
  }
  // Спрятать без записи истории: так лист уходит перед карточкой вина и новым сканом.
  function hide() {
    if (!sheet || !sheet.shown) return;
    abort();
    sheet.shown = false;
    detach(sheet.wrap);
    document.removeEventListener("keydown", sheet.onKey);
    document.body.style.overflow = sheet.overflow || "";
    if (sheet.back && sheet.back.focus) {
      try { sheet.back.focus({ preventScroll: true }); } catch (e) { /* узла уже нет */ }
    }
  }
  function close(opts) {
    if (!sheet || !sheet.shown) return;
    hide();
    if (!(isObj(opts) && opts.fromHistory)) call("onSheet", false);
  }
  function destroy() {
    hide();
    timers.forEach(function (timer) { clearInterval(timer); });
    timers = [];
    sheet = null;
  }
  function isOpen() { return Boolean(sheet && sheet.shown); }

  function setBusy(on) {
    sheet.status.classList.toggle("somm-status--busy", on);
    sheet.statusText.nodeValue = on ? "подбирает ответ" : "на связи";
  }
  function entryChips() {
    var chips = sheet.data ? list(sheet.data.chips) : [];
    return chips.length ? chips : DEFAULT_CHIPS;
  }
  function pickChip(chip) {
    ask({ chip: chip.id, args: isObj(chip.args) ? chip.args : null, text: chip.text });
  }

  // Приветствие, пока вопросов не было: реплика и чипы входа.
  function greet() {
    if (sheet.intro) detach(sheet.intro);
    sheet.intro = h("div", "somm-intro");
    var line = h("div");
    say(line, HELLO);
    add(sheet.intro, line, chipRow(entryChips(), pickChip));
    sheet.feed.insertBefore(sheet.intro, sheet.harm);
  }
  function applyData(data) {
    if (!sheet) return;
    sheet.data = data;
    sheet.winery = sheet.winery || str(data.winery);
    setInput(data.input === true);
    if (sheet.intro && !sheet.turns) greet();
  }
  function loadData(slug) {
    var mine = sheet;
    fetch(WINES + encodeURIComponent(slug) + "/sommelier?order=" + order(), { headers: { Accept: "application/json" } })
      .then(function (response) { return response.ok ? response.json() : null; })
      .catch(function () { return null; })
      .then(function (data) { if (sheet === mine && isObj(data)) applyData(data); });
  }

  // Открыть лист. opts: {name, winery, photo_url, data?, chip?, question?}; chip — объект чипа
  // или его id.
  function open(slug, opts) {
    if (typeof slug !== "string" || !slug) return false;
    opts = isObj(opts) ? opts : {};
    if (sheet && sheet.slug !== slug) destroy();
    var fresh = !sheet;
    if (fresh) build(slug, opts);
    show();
    if (isObj(opts.data) && opts.data.slug === slug) applyData(opts.data);
    else if (fresh) loadData(slug);
    var chip = opts.chip;
    if (typeof chip === "string") chip = { id: chip, text: CHIP_TEXT[chip] || "", args: opts.args };
    if (isObj(chip) && str(chip.id)) pickChip({ id: chip.id, args: chip.args, text: str(chip.text) || CHIP_TEXT[chip.id] || "" });
    else if (typeof opts.question === "string" && clean(opts.question)) ask({ question: clean(opts.question), text: clean(opts.question) });
    else if (!sheet.turns && !sheet.intro) greet();
    else if (sheet.turn && sheet.turn.closed) follow(sheet.turn);   // вернулись после обрыва
    return true;
  }

  // ---------------------------------------------------------------- вопрос и поток
  function ask(req, replay) {
    abort();
    if (sheet.turn && sheet.turn.follow) detach(sheet.turn.follow);   // уточнения — только под последним
    if (sheet.intro) {
      var chips = sheet.intro.lastChild;
      if (chips && chips.className === "somm-chips") detach(chips);
    }
    var turn = { el: h("div", "somm-turn"), stages: h("div", "somm-stages"), body: h("div", "somm-turn__body"),
      tail: h("div", "somm-turn__tail"), stage: null, facts: null, verdict: null, label: null, follow: null,
      closed: false, chips: null };
    if (req.text) turn.el.appendChild(h("p", "somm-q", req.text));
    add(turn.el, turn.stages, turn.body, turn.tail);
    sheet.feed.insertBefore(turn.el, sheet.harm);
    sheet.turn = turn;
    reveal(turn);
    sheet.turns += 1;
    turn.req = req;

    var body = { slug: sheet.slug };
    if (req.question) body.question = req.question;
    else {
      body.chip = req.chip;
      if (req.args) body.args = req.args;
    }
    req.context = replay ? req.context : sheet.context;
    if (req.context) body.context = req.context;

    var ctrl = new AbortController();
    var mine = sheet;
    sheet.ctrl = ctrl;
    setBusy(true);
    fetch(ASK + order(), { method: "POST", signal: ctrl.signal, cache: "no-store",
      headers: { "Content-Type": "application/json", Accept: "application/x-ndjson" }, body: JSON.stringify(body) })
      .then(function (response) {
        if (!response.ok) return refused(turn, response);
        return lines(response, function (event) { handle(turn, event); }).then(function () {
          finish(turn, turn.facts ? null : LOST_TEXT, "lost");   // поток кончился без done
        });
      })
      .catch(function () { if (!ctrl.signal.aborted) finish(turn, turn.facts ? null : LOST_TEXT, "lost"); })
      .then(function () {
        if (sheet === mine && mine.ctrl === ctrl) {
          mine.ctrl = null;
          setBusy(false);
        }
      });
  }
  // Новый ход — вопросом к верху ленты; повторно — когда пришёл ответ и лента выросла.
  function reveal(turn) {
    if (sheet && sheet.turn === turn) sheet.feed.scrollTop = Math.max(turn.el.offsetTop - 12, 0);
  }
  // Оборвать текущий запрос: ответ остаётся шаблоном, незаконченный этап снимается.
  function abort() {
    if (!sheet || !sheet.ctrl) return;
    var ctrl = sheet.ctrl;
    sheet.ctrl = null;
    try { ctrl.abort(); } catch (e) { /* уже закрыт */ }
    if (sheet.turn && !sheet.turn.closed) finish(sheet.turn, sheet.turn.facts ? null : CUT_TEXT, "cut");
    setBusy(false);
  }

  // NDJSON по строкам: fetch + ReadableStream; без потока — весь ответ разом.
  function lines(response, onEvent) {
    var buffer = "";
    function feed(chunk, last) {
      buffer += chunk;
      var parts = buffer.split("\n");
      buffer = last ? "" : parts.pop();
      parts.forEach(function (line) {
        line = line.trim();
        if (!line) return;
        var event = null;
        try { event = JSON.parse(line); } catch (e) { return; }
        if (isObj(event)) onEvent(event);
      });
    }
    if (!response.body || typeof response.body.getReader !== "function" || typeof TextDecoder === "undefined") {
      return response.text().then(function (text) { feed(text, true); });
    }
    var reader = response.body.getReader();
    var decoder = new TextDecoder("utf-8");
    function pump() {
      return reader.read().then(function (step) {
        if (step.done) {
          feed(decoder.decode(), true);
          return null;
        }
        feed(decoder.decode(step.value, { stream: true }), false);
        return pump();
      });
    }
    return pump();
  }

  // Ответ не 200: 404 — вина нет, 422 «вопрос текстом выключен» — поле прячется.
  function refused(turn, response) {
    return response.json().catch(function () { return null; }).then(function (body) {
      var detail = isObj(body) && typeof body.detail === "string" ? body.detail : "";
      if (response.status === 422 && detail === INPUT_OFF) {
        setInput(false);
        finish(turn, INPUT_OFF_TEXT, "lost");
      } else {
        finish(turn, response.status === 404 ? GONE_TEXT : FAIL_TEXT, "lost");
      }
    });
  }

  function handle(turn, event) {
    if (turn.closed) return;
    if (event.type === "stage") stage(turn, event);
    else if (event.type === "facts") facts(turn, event);
    else if (event.type === "text") text(turn, event);
    else if (event.type === "error") finish(turn, str(event.text) || FAIL_TEXT, "lost");
    else if (event.type === "done") finish(turn, turn.facts ? null : FAIL_TEXT, turn.facts ? "done" : "lost");
  }

  // Этап: текущий — с пульсирующей точкой, пройденные — «✓ {done}». Новый закрывает прошлый.
  function closeStage(turn, honest) {
    var now = turn.stage;
    if (!now) return;
    turn.stage = null;
    if (!honest) {
      detach(now.el);   // обрыв: этап не закончен — галочку не ставим
      return;
    }
    clear(now.el);
    now.el.className = "somm-stage somm-stage--done";
    add(now.el, icon("check"), now.done);
  }
  function stage(turn, event) {
    closeStage(turn, true);
    var el = h("span", "somm-stage somm-stage--now");
    add(el, attrs(h("span", "somm-stage__dot"), { "aria-hidden": "true" }), str(event.text));
    turn.stages.appendChild(el);
    turn.stage = { el: el, done: str(event.done) || str(event.text) };
  }

  // Чипы правил блюда: до трёх «+» и до двух «−» (договор §1), каждый — [правило, знак].
  function rulesOf(dish) {
    function pick(rules, n, sign) {
      return list(rules).slice(0, n).filter(function (rule) { return isObj(rule) && str(rule.text); })
        .map(function (rule) { return [rule, sign]; });
    }
    return pick(dish.plus, 3, "+").concat(pick(dish.minus, 2, "−"));
  }
  // Источник у всех чипов блюда один — пишется раз, в строке блюда, а не «· по сорту» у каждого.
  // Считается по самим чипам: нет у них источника (ответ другой версии) — пометок нет вовсе.
  function dishSource(dish) {
    var seen = rulesOf(dish).map(function (item) { return item[0].source; });
    return seen.length && seen.every(function (s) { return s === seen[0]; }) && own(RULE_SOURCE, seen[0]) ? seen[0] : "";
  }
  function ruleChips(dish, shared) {
    var row = h("div", "somm-rules");
    rulesOf(dish).forEach(function (item) {
      var src = shared ? "" : own(RULE_SOURCE, item[0].source);
      row.appendChild(add(h("span", "somm-rule somm-rule--" + (item[1] === "+" ? "plus" : "minus"), item[1] + " " + item[0].text),
        src ? h("span", "somm-rule__src", " · " + src) : null));
    });
    return row.firstChild ? row : null;
  }
  function dishList(dishes) {
    var box = add(h("section", "somm-dishes"), h("p", "somm-eyebrow", "Блюда и правила сочетаний"));
    dishes.forEach(function (dish) {
      if (!isObj(dish) || !str(dish.name)) return;
      var src = dishSource(dish);
      var said = [own(VERDICT, dish.verdict), own(RULE_SOURCE, src)].filter(Boolean).join(" · ");
      box.appendChild(add(h("div", "somm-dish"),
        add(h("p", "somm-dish__head"), h("span", "somm-dish__name", dish.name), said ? h("span", "somm-dish__verdict", said) : null),
        ruleChips(dish, src)));
    });
    return box;
  }

  // Плашка 149-ФЗ — одна на лист, над первой подборкой; дальше — значок (i). При обычной
  // сортировке на её месте та же плашка с «Обычная сортировка: по названию» и включённым
  // переключателем: он возвращает рекомендации, не уходя из листа. Текст — о подборке под
  // плашкой и после переключения не меняется: подборка осталась какой была, и рекомендательная
  // не теряет плашку 149-ФЗ; в тон сортировке встаёт только переключатель.
  function notice(text, plain) {
    // Состояние «вкл/выкл» — у самого checkbox: aria-checked на нём не ставится (ARIA in HTML).
    var input = attrs(h("input", "somm-switch__input"), { type: "checkbox", role: "switch" });
    (sheet.paint = function () { input.checked = order() === "plain"; })();
    input.addEventListener("change", function () {
      call("setOrder", input.checked ? "plain" : "reco");
      if (sheet && sheet.winesReq) ask(sheet.winesReq, true);   // та же подборка в новой сортировке
    });
    return add(h("div", "somm-notice"), add(h("span", "somm-notice__text"), icon("info"), plain ? PLAIN_NOTE : text || NOTICE),
      add(h("label", "somm-switch"), input, attrs(h("span", "somm-switch__track"), { "aria-hidden": "true" }), "Обычная сортировка"));
  }
  function infoMark(text) {
    return add(attrs(h("span", "somm-info"), { role: "img", "aria-label": text, title: text }), icon("info"));
  }

  function compareOf(raw) {
    if (!isObj(raw) || !isObj(raw.anchor) || !Array.isArray(raw.anchor.profile)) return null;
    var byWine = {};
    list(raw.wines).forEach(function (wine) {
      if (isObj(wine) && str(wine.slug) && Array.isArray(wine.profile)) byWine[wine.slug] = wine.profile;
    });
    return { anchor: raw.anchor, name: str(raw.anchor.name), byWine: byWine, drawn: {},
      axes: list(raw.axes).filter(function (axis) { return AXES.indexOf(axis) >= 0; }) };
  }
  // Мини-шкала под плиткой: якорь — полая --orig, предложенное — как у шкал карточки.
  function miniScale(axis, anchor, profile, drawn) {
    var a = axisOf(anchor, axis), b = axisOf(profile, axis);
    if (!a || !b) return null;
    drawn[b.source] = true;
    var weak = edgeOf(a, b);
    return add(h("span", "somm-mini"), h("span", "somm-mini__label", b.label + (weak ? " (" + SOURCE_TEXT[weak] + ")" : "")),
      track(b, a, "somm-track--mini"));
  }
  function tile(wine, compare) {
    var el = button("somm-tile");
    var meta = [str(wine.winery), str(wine.region)].filter(Boolean).join(" · ");
    var text = add(h("span", "somm-tile__body"), h("span", "somm-tile__name", str(wine.name) || wine.slug),
      meta ? h("span", "somm-tile__meta", meta) : null, str(wine.pill) ? h("span", "somm-pill", wine.pill) : null);
    if (compare && compare.byWine[wine.slug]) {
      compare.axes.forEach(function (axis) {
        add(text, miniScale(axis, compare.anchor.profile, compare.byWine[wine.slug], compare.drawn));
      });
    }
    add(el, photo(wine.photo_url, "somm-tile__photo somm-tint--" + tint(wine.style_label)), text);
    el.addEventListener("click", function () {
      hide();
      call("openWine", wine.slug);
    });
    return el;
  }
  function selection(f, wines) {
    var box = h("section", "somm-wines");
    var note149 = str(f.notice_149);
    if ((note149 || f.order === "plain") && !sheet.notice) {
      sheet.notice = true;
      box.appendChild(notice(note149, f.order === "plain"));
      note149 = "";
    }
    var compare = compareOf(f.compare);
    var head = add(h("div", "somm-wines__head"), h("span", "somm-eyebrow", str(f.wines_title) || "Вина других виноделен"),
      note149 ? infoMark(note149) : null);
    var tiles = h("div", "somm-tiles");
    wines.forEach(function (wine) { if (isObj(wine) && str(wine.slug)) tiles.appendChild(tile(wine, compare)); });
    add(box, head, tiles);
    // Мини-шкалы: полая серая точка — исходное вино, точка акцента — предложенное (сплошная или
    // полая — как нарисована).
    if (compare && Object.keys(compare.drawn).length) {
      head.appendChild(add(h("span", "somm-legend"), mark("orig", anchorName(compare.name) + " — исходное"),
        mark(["catalog", "grape", "style"].filter(function (kind) { return compare.drawn[kind]; }), "предложенное")));
    }
    // «Чем заменить»: мини-шкал нет — «розой ветров» поверх исходного вино compare.overlay (сильнее
    // всех отличное; null — все совпали, наложения нет), в ответе без поля — первое подборки.
    var picked = isObj(f.compare) && Object.prototype.hasOwnProperty.call(f.compare, "overlay");
    var first = !picked ? wines[0] : wines.filter(function (wine) { return isObj(wine) && wine.slug === f.compare.overlay; })[0];
    if (compare && !compare.axes.length && isObj(first) && compare.byWine[first.slug]) {
      var pic = h("div");
      var own = compare.byWine[first.slug];
      if (radar(pic, own, { profile: compare.anchor.profile, name: anchorName(compare.name) })) {
        var kinds = kindsOf(slotsOf(own, compare.anchor.profile));
        var firstName = [str(first.name) || "предложенное", str(first.winery)].filter(Boolean).join(" · ");
        box.appendChild(add(h("figure", "somm-overlay"),
          h("figcaption", "somm-eyebrow", "Профиль рядом с исходным"), pic,
          add.apply(null, [h("p", "somm-legend somm-legend--rows"), mark("orig-line", anchorName(compare.name) + " — исходное"),
            mark("own-line", firstName)].concat(sourceMarks(kinds)))));
      }
    }
    return box;
  }
  // Исходное вино — с винодельней: у предложенных название часто то же («Саперави»).
  function anchorName(name) {
    return [str(name) || sheet.name, sheet.winery].filter(Boolean).join(" · ");
  }

  function facts(turn, f) {
    turn.facts = f;
    if (isObj(f.context)) sheet.context = f.context;
    var head = add(h("div", "somm-answer__head"), h("span", "somm-eyebrow", "Коротко"));
    var basis = list(f.basis).filter(function (item) { return typeof item === "string" && item; }).join(" · ");
    if (basis) head.appendChild(h("span", "somm-answer__basis", basis));
    turn.verdict = h("p", "somm-verdict", nb(f.verdict_template));
    var block = add(h("div", "somm-answer"), head, turn.verdict);
    if (str(f.detail)) block.appendChild(h("p", "somm-answer__detail", nb(f.detail)));
    var dishes = list(f.dishes);
    var one = dishes.length === 1 && isObj(dishes[0]) ? dishes[0] : null, src = one ? dishSource(one) : "";
    if (one) add(block, ruleChips(one, src), src ? h("p", "somm-rules__src", "Правила сочетаний — " + RULE_SOURCE[src]) : null);
    turn.body.appendChild(block);
    if (dishes.length > 1) turn.body.appendChild(dishList(dishes));
    var wines = list(f.wines);
    if (wines.length) {
      sheet.winesReq = turn.req;
      turn.body.appendChild(selection(f, wines));
    }
    turn.chips = list(f.chips);
    turn.label = labelLine(false);   // шаблон виден сразу с меткой алгоритма
    turn.tail.appendChild(turn.label);
    reveal(turn);
  }

  // Итоговый текст: проверенный текст модели печатается кареткой, шаблон остаётся как есть.
  function text(turn, t) {
    var ai = t.generated === true && t.label === LABEL_AI && str(t.text) !== "";
    if (!turn.verdict) {
      turn.verdict = h("p", "somm-verdict");
      turn.body.appendChild(add(h("div", "somm-answer"), turn.verdict));
    }
    var fresh = labelLine(ai);
    if (turn.label) turn.tail.replaceChild(fresh, turn.label);
    else turn.tail.appendChild(fresh);
    turn.label = fresh;
    if (ai) type(turn.verdict, nb(t.text));
    else if (str(t.text) && nb(t.text) !== turn.verdict.textContent) turn.verdict.textContent = nb(t.text);
  }
  // Каретка: 18 мс на символ; невидимый остаток держит высоту абзаца — вёрстка не прыгает.
  function type(el, full) {
    if (reducedMotion()) {
      el.textContent = full;
      return;
    }
    clear(el);
    el.setAttribute("aria-busy", "true");
    var shown = document.createTextNode("");
    var rest = attrs(h("span", "somm-ghost", full), { "aria-hidden": "true" });
    add(el, shown, attrs(h("span", "somm-caret"), { "aria-hidden": "true" }), rest);
    var at = 0;
    var timer = setInterval(function () {
      at += 1;
      shown.nodeValue = full.slice(0, at);
      rest.textContent = full.slice(at);
      if (at < full.length) return;
      clearInterval(timer);
      timers = timers.filter(function (other) { return other !== timer; });
      el.textContent = full;
      el.removeAttribute("aria-busy");
    }, CARET_MS);
    timers.push(timer);
  }

  // Конец хода. how: done — сервер закрыл все этапы; lost — поток оборвался или пришла ошибка:
  // незаконченный этап снимается, ниже — текст и чипы входа; cut — человек сам оборвал ответ
  // (новый вопрос, закрытый лист): уточнений нет, их даст следующий ход или повторное открытие.
  function finish(turn, problem, how) {
    if (turn.closed) return;
    turn.closed = true;
    closeStage(turn, how === "done");
    if (problem) {
      turn.body.appendChild(add(h("div", "somm-answer somm-answer--error"), h("p", "somm-answer__error", problem)));
      turn.chips = entryChips();
    }
    if (how !== "cut") follow(turn);
  }
  function follow(turn) {
    if (!sheet || sheet.turn !== turn || turn.follow) return;
    turn.follow = chipRow(list(turn.chips).length ? turn.chips : entryChips(), pickChip);
    if (turn.follow) turn.tail.insertBefore(turn.follow, turn.tail.firstChild);
  }

  // ---------------------------------------------------------------- вход
  function init(page) {
    if (isObj(page)) {
      Object.keys(host).forEach(function (key) { if (typeof page[key] === "function") host[key] = page[key]; });
    }
    // Уход со страницы обрывает поток: сервер снимает голос и больше ничего не шлёт. Уточнения
    // остаются под последним ходом: страница может вернуться из кэша «назад/вперёд» с листом.
    window.addEventListener("pagehide", function () {
      if (!sheet) return;
      abort();
      if (sheet.turn) follow(sheet.turn);
    });
  }

  window.SommUI = {
    version: 1,
    init: init,
    note: note,
    radar: radar,
    scales: scales,
    entry: entry,
    open: open,
    close: close,
    isOpen: isOpen,
    say: say
  };
})();
