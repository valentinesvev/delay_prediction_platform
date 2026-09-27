/* Live dispatcher: all vehicle data comes from the read-only FastAPI fleet endpoint. */
(async () => {
  'use strict';
  const Data = window.DispatcherData;
  let snapshot = {time:'—', vehicles:[]}, metadata = {}, connected = false, loading = false, pollTimer = null;
  const $ = id => document.getElementById(id);
  const state = {selected:null, filter:'all', query:'', mapStyle:'positron', gps:true, stops:true, segmentIndex:null, scope:'all', threshold:300};
  let map = null, vehicleLayer = null, trackLayer = null, segmentAnalyticsLayer = null, tiles = null, styleLayer = null;
  const issueColors = { normal:'#a7a4a0', warning:'#e5a32b', critical:'#ca2437', blocked:'#4d2630' };
  const icon = '<svg aria-hidden="true"><use href="#bus-icon"/></svg>';
  const escapeHTML = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const frame = () => snapshot;
  const delayText = seconds => seconds == null ? '—' : `${seconds > 0 ? '+' : seconds < 0 ? '−' : ''}${new Intl.NumberFormat('ru-RU', { maximumFractionDigits: 1 }).format(Math.abs(seconds) / 60)} мин`;
  const inScope = v => state.scope === 'all' || state.scope === 'east' && v.position[1] >= 37.7 || state.scope === 'south' && v.position[0] < 55.7;
  const scopedVehicles = () => frame().vehicles.filter(inScope);
  const signal = v => Data.signal(v, state.threshold, connected);
  const fresh = v => Data.fresh(v, connected);
  const status = v => {
    if (!connected) return {label:'Нет связи · данные устарели', tone:'gray'};
    if (v.age > 120) return {label:'Координаты устарели', tone:'gray'};
    const s = signal(v);
    if (s.kind === 'late') return {label:'Текущая задержка ≥ порога', tone:'red'};
    if (s.kind === 'risk') return {label:'Прогноз задержки ≥ порога', tone:'amber'};
    if (v.forecast && !fresh(v)) return {label:'Прогноз устарел', tone:'gray'};
    if (v.routeStatus === 'off_route') return {label:'Далеко от плана', tone:'gray'};
    if (!v.hasSchedule) return {label:'Нет плана', tone:'gray'};
    if (!v.forecast) return {label:'Ожидаем прогноз', tone:'gray'};
    return {label:'Ниже порога', tone:'green'};
  };
  const ageText = seconds => seconds < 60 ? `${seconds} с назад` : `${Math.floor(seconds / 60)} мин назад`;
  const workspace = document.querySelector('.workspace');
  const mapStage = document.querySelector('.map-stage');
  let cardsWereHidden = false;

  function resizeMap() {
    requestAnimationFrame(() => map?.invalidateSize({pan:false}));
  }

  function setSidebarOpen(open) {
    workspace.classList.toggle('sidebar-collapsed', !open);
    $('transport-sidebar').hidden = !open;
    $('sidebar-toggle').setAttribute('aria-expanded', String(open));
    $('sidebar-open').setAttribute('aria-expanded', String(open));
    $('sidebar-open').hidden = open;
    resizeMap();
  }

  function setCleanMap(clean) {
    mapStage.classList.toggle('clean-map', clean);
    $('map-cards-toggle').setAttribute('aria-pressed', String(clean));
    $('map-cards-toggle').textContent = clean ? 'Показать карточки' : 'Скрыть карточки';
  }

  function setMapExpanded(expanded) {
    if (expanded) {
      cardsWereHidden = mapStage.classList.contains('clean-map');
      setCleanMap(true);
    } else {
      setCleanMap(cardsWereHidden);
    }
    workspace.classList.toggle('map-expanded', expanded);
    document.body.classList.toggle('map-focus', expanded);
    $('map-expand-toggle').setAttribute('aria-pressed', String(expanded));
    $('map-expand-toggle').textContent = expanded ? 'Свернуть карту' : 'Развернуть карту';
    resizeMap();
  }

  function mapMessage(text) {
    $('map-message').textContent = text;
    $('map-message').hidden = !text;
  }

  function setMapStyle(style) {
    state.mapStyle = style;
    $('map').className = `map-theme-${style}`;
    document.querySelectorAll('[data-map-style]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.mapStyle === style)));
    if (!map) return;
    if (!L.maplibreGL) {
      const fallbackFilters = { positron:'grayscale(.78) brightness(1.08)', bright:'none' };
      tiles.getContainer().style.filter = fallbackFilters[style];
      return;
    }
    if (!map.hasLayer(tiles)) tiles.addTo(map);
    if (styleLayer) map.removeLayer(styleLayer);
    styleLayer = L.maplibreGL({ style: `https://tiles.openfreemap.org/styles/${style}` }).addTo(map);
    const currentLayer = styleLayer;
    const glMap = styleLayer.getMaplibreMap?.();
    glMap?.on('load', () => { if (currentLayer === styleLayer && map.hasLayer(tiles)) map.removeLayer(tiles); mapMessage(''); });
    glMap?.on('error', () => { if (currentLayer === styleLayer) mapMessage('Векторная карта загружается с ошибкой. Доступна резервная карта OpenStreetMap.'); });
  }

  function setupMap() {
    if (!window.L) {
      mapMessage('Карта не загрузилась. Проверьте подключение к интернету и обновите страницу. Данные транспорта доступны в списке.');
      $('overview-button').disabled = true;
      return;
    }
    map = L.map('map', { zoomControl: false, scrollWheelZoom: true }).setView([55.7512, 37.6184], 11);
    L.control.zoom({ position: 'topright' }).addTo(map);
    tiles = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a> contributors' }).addTo(map);
    let tileErrors = false;
    tiles.on('loading', () => { tileErrors = false; });
    tiles.on('tileerror', () => {
      tileErrors = true;
      mapMessage('Часть карты не загрузилась. Проверьте интернет. Список автобусов остаётся доступен.');
    });
    tiles.on('load', () => { if (!tileErrors) mapMessage(''); });
    vehicleLayer = L.layerGroup().addTo(map);
    trackLayer = L.layerGroup().addTo(map);
    segmentAnalyticsLayer = L.layerGroup().addTo(map);
    setMapStyle(state.mapStyle);
  }

  window.DispatcherSegmentMap = {
    update(rows, selectedRoute) {
      if (!map || !segmentAnalyticsLayer) return;
      segmentAnalyticsLayer.clearLayers();
      const visible = selectedRoute ? rows.filter(row => row.route_id === selectedRoute) : rows.filter(row => row.severity !== 'NORMAL');
      visible.forEach(row => {
        const color = row.severity === 'CRITICAL' || row.severity === 'WARNING' ? '#c92039' : row.severity === 'WATCH' ? '#e5a32b' : '#547b80';
        const coords = [[row.from_lat, row.from_lon], [row.to_lat, row.to_lon]];
        const label = document.createElement('span');
        label.textContent = `${row.route_name}: ${row.from_name} → ${row.to_name} · ${row.severity}`;
        L.polyline(coords, {color:'#fff', weight:12, opacity:.95, interactive:false}).addTo(segmentAnalyticsLayer);
        L.polyline(coords, {color, weight:7, opacity:.95}).bindTooltip(label).addTo(segmentAnalyticsLayer);
        if (row.severity === 'WARNING' || row.severity === 'CRITICAL') {
          const center = [(row.from_lat + row.to_lat)/2, (row.from_lon + row.to_lon)/2];
          L.marker(center, {icon:L.divIcon({html:'<span class="issue-marker critical">!</span>',className:'issue-pin',iconSize:[34,34],iconAnchor:[17,17]}),title:label.textContent}).bindTooltip(label).addTo(segmentAnalyticsLayer);
        }
      });
    },
    focus(row) {
      if (map) map.fitBounds(L.latLngBounds([[row.from_lat,row.from_lon],[row.to_lat,row.to_lon]]), {padding:[90,90],maxZoom:15,animate:false});
      mapStage.scrollIntoView({behavior:'smooth',block:'center'});
    }
  };

  function routeSegments(v) {
    if (!v?.stops?.length) return [];
    const points = v.age > 120 ? [...v.stops] : [{ position: v.position, address: 'Положение автобуса', time: frame().time }, ...v.stops];
    const segments = points.slice(1).map((stop, index) => ({ index, from: points[index], to: stop }));
    if (!segments.length) return segments;
    const targetIndex = v.stops.findIndex(stop => stop.position[0] === v.forecast?.target?.[0] && stop.position[1] === v.forecast?.target?.[1]);
    const incidentIndex = Math.min(segments.length - 1, Math.max(0, targetIndex - (v.age > 120 ? 1 : 0)));
    let incident = null;
    const sig = signal(v);
    if (sig.kind === 'late') {
      incident = {tone:'critical', label:'Оценка задержки ≥ порога', note:`${delayText(v.currentDelay)} по GPS и плану · причина неизвестна`};
    } else if (sig.kind === 'risk' && targetIndex >= 0) {
      incident = {tone:'warning', label:'Прогноз задержки ≥ порога', note:`${delayText(v.forecast.delay)} к целевой остановке · это не вероятность сбоя`};
    }
    return segments.map(segment => ({...segment, issue:segment.index === incidentIndex ? incident : null, color:segment.index === incidentIndex && incident ? issueColors[incident.tone] : issueColors.normal }));
  }

  function renderSegments() {
    const v = frame().vehicles.find(item => item.id === state.selected);
    const latest = v?.track?.at(-1);
    const lastPair = latest?.length >= 2 ? latest.slice(-2) : null;
    const currentColor = signal(v).kind === 'late' ? '#ca2437' : signal(v).kind === 'risk' ? '#e5a32b' : '#4e1930';
    $('current-gps-segment').style.setProperty('--current-color', currentColor);
    $('current-gps-segment').innerHTML = lastPair && v.age <= 120
      ? `<span class="gps-now-badge">СЕЙЧАС / GPS</span><strong>Текущий отрезок движения автобуса ${escapeHTML(v.id)}</strong><span>Последние две записанные точки · ${escapeHTML(v.lastSeen)} · ${escapeHTML(v.speed)} км/ч</span><small>Название улицы и границы дорожного участка в данных не указаны. Отрезок выделен на карте.</small>`
      : `<span class="gps-now-badge">СЕЙЧАС / GPS</span><strong>Текущий дорожный отрезок не определён</strong><small>${v?.age > 120 ? 'Координаты устарели.' : 'Недостаточно последовательных GPS-точек.'}</small>`;
    const segments = routeSegments(v);
    if (!segments.length) {
      $('segment-list').innerHTML = '<p class="segment-empty">Для этого автобуса нет ближайших точек расписания. Можно посмотреть записанный GPS‑путь на карте.</p>';
      return;
    }
    $('segment-list').innerHTML = segments.map(s => `<button type="button" class="segment-card ${s.issue ? `segment-${s.issue.tone}` : ''}" data-segment-index="${s.index}" aria-pressed="${s.index === state.segmentIndex}" style="--segment-color:${s.color}"><span class="segment-number">${String(s.index + 1).padStart(2, '0')}</span><span class="segment-line"><i></i><i></i></span><strong>${escapeHTML(s.from.address)} → ${escapeHTML(s.to.address)}</strong><small>${escapeHTML(s.from.time)} — ${escapeHTML(s.to.time)} · ${s.issue ? escapeHTML(s.issue.label) : 'без сигнала'}</small></button>`).join('');
  }

  function renderIncident() {
    const v = frame().vehicles.find(item => item.id === state.selected);
    const segment = routeSegments(v).find(item => item.issue);
    const banner = $('incident-banner');
    if (!segment) {
      banner.className = 'incident-banner incident-none';
      banner.disabled = true;
      banner.innerHTML = '<strong>Нет выделенного проблемного участка</strong><span>Нет актуального сигнала для выбранного транспорта.</span>';
      return;
    }
    banner.className = `incident-banner incident-${segment.issue.tone}`;
    banner.disabled = false;
    banner.innerHTML = `<span class="incident-symbol">${segment.issue.tone === 'blocked' ? '×' : '!'}</span><span><strong>${escapeHTML(segment.issue.label)}</strong><small>${escapeHTML(segment.from.address)} → ${escapeHTML(segment.to.address)}</small><em>${escapeHTML(segment.issue.note)}</em></span><b aria-hidden="true">↗</b>`;
  }

  function renderVerdict() {
    const vehicles = scopedVehicles();
    const late = vehicles.filter(v => signal(v).kind === 'late').length;
    const risk = vehicles.filter(v => signal(v).kind === 'risk').length;
    const count = vehicles.filter(fresh).length;
    const tone = !connected ? 'normal' : late ? 'critical' : risk ? 'warning' : 'normal';
    const title = !connected ? 'Ожидаем соединение с сервером' : !vehicles.length ? 'Нет свежей телеметрии' : late ? 'Есть транспорт с текущей задержкой' : risk ? 'Прогнозируется задержка выше порога' : 'Нет сигналов выше порога';
    $('verdict-panel').className = `verdict-panel verdict-${tone}`;
    $('verdict-panel').innerHTML = `<div class="verdict-main"><span class="verdict-overline">ОБЗОР / ${metadata.mode === 'historical' ? 'ИСТОРИЧЕСКАЯ БАЗА' : 'ДАННЫЕ БАЗЫ'}</span><strong>${title}</strong><p>Порог: ${state.threshold / 60} мин. Текущая задержка — оценка по GPS; прогноз — сохранённый результат модели. Причины и вероятность сбоя не определяются.</p></div><div class="verdict-stats"><div><b>${connected ? late : '—'}</b><span>Текущая задержка<br>≥ порога</span></div><div><b>${connected ? risk : '—'}</b><span>Прогноз<br>≥ порога</span></div><div><b>${count}</b><span>Свежих<br>прогнозов</span></div></div>`;
  }

  function visibleVehicles() {
    const q = state.query.trim().toLocaleLowerCase('ru');
    return scopedVehicles().filter(v => (state.filter === 'all' || v.forecast || signal(v).kind !== 'none') && (!q || v.id.includes(q) || (v.forecast?.address || '').toLocaleLowerCase('ru').includes(q) || v.stops.some(s => s.address.toLocaleLowerCase('ru').includes(q))))
      .sort((a, b) => ({blocked:4,late:3,risk:2,none:1}[signal(b).kind] - {blocked:4,late:3,risk:2,none:1}[signal(a).kind]) || (b.forecast?.delay ?? -Infinity) - (a.forecast?.delay ?? -Infinity) || a.id.localeCompare(b.id));
  }

  function renderList() {
    const vehicles = visibleVehicles();
    $('forecast-count').textContent = scopedVehicles().filter(v => v.forecast).length;
    $('all-count').textContent = scopedVehicles().length;
    $('stat-vehicles').textContent = scopedVehicles().length;
    $('stat-forecasts').textContent = scopedVehicles().filter(v => v.forecast).length;
    $('count-label').textContent = `${scopedVehicles().length} автобусов в зоне · ${frame().time}`;
    document.querySelectorAll('[data-filter]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.filter === state.filter)));
    if (!vehicles.length) {
      $('vehicle-list').innerHTML = '<p class="empty-list">Ничего не найдено. Попробуйте другой ID или адрес.</p>';
      return;
    }
    $('vehicle-list').innerHTML = vehicles.map(v => {
      const s = status(v), f = v.forecast;
      const address = f?.address || v.stops[0]?.address || (v.hasSchedule ? 'Нет ближайших точек расписания' : 'Расписание отсутствует');
      const sig = signal(v);
      return `<button type="button" class="vehicle-card signal-${sig.kind}" data-vehicle="${escapeHTML(v.id)}" aria-pressed="${v.id === state.selected}" aria-controls="vehicle-detail"><div class="vehicle-top"><span class="bus-mini">${icon}</span><div><div class="vehicle-name">Автобус ${escapeHTML(v.id)}</div><div class="vehicle-caption">${f ? 'Прогноз из базы · ' + escapeHTML(f.model) : 'Данные телеметрии'}</div></div></div><p class="vehicle-address">${escapeHTML(address)}</p><span class="status-chip ${s.tone}">${s.label}</span><div class="card-bottom" style="margin-top:14px"><span>${f ? `К ${f.plan} МСК` : `Координаты: ${v.lastSeen}`}</span><strong>${f ? delayText(f.delay) : '—'}</strong></div></button>`;
    }).join('');
  }

  function renderDetails() {
    const v = frame().vehicles.find(v => v.id === state.selected);
    if (!v) { $('vehicle-detail').innerHTML = '<p>Нет выбранного транспорта. Ожидаем телеметрию или выберите другую зону.</p>'; return; }
    const s = status(v), f = v.forecast;
    let body = `<div class="detail-top"><strong>ТС ${escapeHTML(v.id)}</strong><span class="status-chip ${s.tone}">${s.label}</span></div>`;
    const nearby = v.currentStop || v.nextStop;
    body += `<p class="detail-sub">${v.currentStop ? 'Рядом с остановкой' : 'Следующая остановка'}: ${escapeHTML(nearby?.address || 'нет данных')}</p>`;
    body += `<div class="arrival-grid"><div><span>Задержка сейчас ≈</span><b>${delayText(v.currentDelay)}</b></div><div><span>Прогноз задержки</span><b>${delayText(f?.delay)}</b></div><div><span>Ожидаемое прибытие</span><b>${f?.arrival || '—'}</b></div></div>`;
    if (f) {
      body += `<p class="detail-address">${escapeHTML(f.address)}</p><p class="detail-sub">План: ${f.plan} МСК · рассчитано: ${f.calculated} МСК · ${escapeHTML(f.model)}</p>`;
      if (!fresh(v)) body += '<p class="detail-warning">Прогноз устарел или связь с сервером потеряна. Он показан справочно.</p>';
      if (f.degraded) body += '<p class="detail-warning">Прогноз рассчитан при неполной телеметрии.</p>';
      if (f.reason || f.model === 'baseline') body += `<p class="detail-warning">Резервный расчёт: ${escapeHTML(f.reason || 'baseline')}</p>`;
      if (f.interval.every(Number.isFinite)) body += `<p class="detail-sub">Интервал задержки: ${delayText(f.interval[0])} … ${delayText(f.interval[1])}</p>`;
    } else {
      body += '<p class="detail-empty">Прогноза пока нет: ожидаем воркер или плановую остановку в окне +10…15 минут. Транспорт показан по телеметрии.</p>';
    }
    body += `<div class="detail-foot"><span>Координаты: ${ageText(v.age)}</span><span>Скорость: ${v.speed == null ? '—' : escapeHTML(v.speed) + ' км/ч'} · МСК</span></div>`;
    $('vehicle-detail').innerHTML = body;
  }

  function drawMap(focus = false) {
    if (!map) return;
    vehicleLayer.clearLayers(); trackLayer.clearLayers();
    scopedVehicles().forEach(v => {
      const selected = v.id === state.selected;
      const sig = signal(v);
      const html = `<div class="bus-marker signal-${sig.kind} ${selected ? 'selected' : ''} ${!connected || v.age > 120 ? 'old' : ''}">${icon}</div>`;
      const marker = L.marker(v.position, { icon: L.divIcon({html,className:'bus-pin',iconSize:selected?[46,46]:[40,40],iconAnchor:selected?[23,23]:[20,20]}),title:`Автобус ${v.id}${v.age > 120 ? ', координаты устарели' : ''}`,alt:`Автобус ${v.id}`,zIndexOffset:selected?1000:0 }).addTo(vehicleLayer);
      marker.on('click', () => choose(v.id));
    });
    const v = frame().vehicles.find(v => v.id === state.selected);
    if (!v) return;
    if (state.gps) v.track.forEach(segment => {
      L.polyline(segment, {color:'#fff',weight:11,opacity:.98,interactive:false}).addTo(trackLayer);
      L.polyline(segment, {color:'#ac1832',weight:6,opacity:.98,interactive:false}).addTo(trackLayer);
    });
    const latest = v.track?.at(-1);
    if (state.gps && v.age <= 120 && latest?.length >= 2) {
      const pair = latest.slice(-2);
      L.polyline(pair,{color:'#fff',weight:17,opacity:1,interactive:false}).addTo(trackLayer);
      const currentColor = signal(v).kind === 'late' ? '#ca2437' : signal(v).kind === 'risk' ? '#e5a32b' : '#4e1930';
      L.polyline(pair,{color:currentColor,weight:11,opacity:1,interactive:false}).bindTooltip('Текущий GPS-отрезок').addTo(trackLayer);
    }
    if (state.stops) {
      routeSegments(v).forEach(s => {
        const active = s.index === state.segmentIndex;
        L.polyline([s.from.position, s.to.position], {color:'#fff',weight:s.issue || active?13:8,opacity:.96,interactive:false}).addTo(trackLayer);
        L.polyline([s.from.position, s.to.position], {color:s.color,weight:s.issue || active?8:4,opacity:active?1:.9,dashArray:s.issue?'10 5':'5 9',interactive:false}).addTo(trackLayer);
        if (s.issue) {
          const midpoint = [(s.from.position[0] + s.to.position[0]) / 2, (s.from.position[1] + s.to.position[1]) / 2];
          const sign = `<span class="issue-marker ${s.issue.tone}">!</span>`;
          L.marker(midpoint,{icon:L.divIcon({html:sign,className:'issue-pin',iconSize:[34,34],iconAnchor:[17,17]}),title:s.issue.label,zIndexOffset:1500}).bindTooltip(s.issue.label).addTo(trackLayer);
        }
      });
      v.stops.forEach((s, index) => {
        const html = `<span class="stop-marker">${index + 1}</span>`;
        L.marker(s.position, {icon:L.divIcon({html,className:'stop-pin',iconSize:[26,26],iconAnchor:[13,13]}),title:`Точка расписания ${index + 1}: ${s.address}, ${s.time}`}).bindTooltip(`${escapeHTML(s.address)} · ${escapeHTML(s.time)}`).addTo(trackLayer);
      });
    }
    if (v.forecast?.target) {
      const tooltip = document.createElement('div');
      tooltip.textContent = `${v.forecast.address} · прогноз ${delayText(v.forecast.delay)}${fresh(v) ? '' : ' (устарел)'} к ${v.forecast.plan}`;
      L.circleMarker(v.forecast.target,{radius:10,color:'#dc3447',weight:3,fillColor:'#fff',fillOpacity:1}).bindTooltip(tooltip,{permanent:true,direction:'bottom',offset:[0,12]}).addTo(trackLayer);
    }
    if (focus) {
      const issue = routeSegments(v).find(segment => segment.issue);
      const coords = issue ? [issue.from.position, issue.to.position] : [v.position, ...v.stops.map(s => s.position)];
      map.fitBounds(L.latLngBounds(coords), {paddingTopLeft:[45,175],paddingBottomRight:[55,215],maxZoom:15,animate:false});
    }
  }

  function render(focus = false) {
    $('current-time').textContent = frame().time;
    $('current-date').textContent = Data.date(metadata.reference_time) + ' · МСК';
    renderList(); renderDetails(); renderSegments(); renderIncident(); renderVerdict(); drawMap(focus);
  }

  function choose(id) {
    state.selected = id;
    state.segmentIndex = null;
    render(true);
    $('announcement').textContent = `Выбран автобус ${id}. Подробности прогноза обновлены.`;
  }

  function selectAvailable() {
    if (!scopedVehicles().some(v => v.id === state.selected)) {
      state.selected = scopedVehicles()[0]?.id || null;
      state.segmentIndex = null;
    }
  }

  async function refresh() {
    if (loading) return;
    loading = true;
    clearTimeout(pollTimer);
    $('refresh-button').disabled = true;
    const badge = $('server-status');
    try {
      const response = await fetch('/vehicles/active', {cache:'no-store', signal:AbortSignal.timeout(10000)});
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      const next = Data.adapt(data);
      const first = !snapshot.vehicles.length && next.vehicles.length;
      snapshot = next; metadata = data; connected = true;
      selectAvailable();
      state.segmentIndex = null;
      const historical = data.mode === 'historical';
      $('mode-label').textContent = data.replay ? 'ИСТОРИЧЕСКОЕ ВОСПРОИЗВЕДЕНИЕ' : historical ? 'ИСТОРИЧЕСКАЯ БАЗА' : 'ЖИВОЙ ПОТОК';
      $('score-source').textContent = historical ? 'СОСТОЯНИЕ НА ВРЕМЯ ТЕЛЕМЕТРИИ' : 'ТЕЛЕМЕТРИЯ · FASTAPI';
      $('data-source').textContent = (data.replay ? 'Историческое воспроизведение: время карты, прогнозов и аналитики движется вместе. ' : historical ? 'Исторический режим: состояние на момент последней точки базы. ' : 'Живые данные: транспорт с достоверной точкой за последние 2 минуты. ') + (data.demo_plan ? 'ДЕМО: искусственное расписание и случайное движение эмулятора; это не проверка точности модели.' : 'API читает сохранённые прогнозы из базы результатов.');
      const ready = snapshot.vehicles.filter(fresh).length;
      badge.className = `server-status server-${ready ? 'ready' : 'unavailable'}`;
      badge.textContent = `FastAPI подключён · ТС: ${snapshot.vehicles.length} · свежих прогнозов: ${ready}`;
      $('last-update').textContent = 'Получено ' + Data.clock(new Date().toISOString()) + ' МСК';
      render(Boolean(first));
    } catch {
      connected = false;
      badge.className = 'server-status server-error';
      badge.textContent = 'Нет актуального ответа FastAPI. Показанные данные устарели; повторяем запрос через 5 секунд.';
      $('mode-label').textContent = 'НЕТ СВЯЗИ';
      render();
    } finally {
      loading = false;
      $('refresh-button').disabled = false;
      pollTimer = setTimeout(refresh, 5000);
    }
  }
  try {
    setupMap();
  } catch {
    map = null;
    mapMessage('Карта недоступна. Телеметрия и прогнозы остаются доступны в списке.');
  }
  render(); refresh();
  $('refresh-button').addEventListener('click', refresh);
  $('vehicle-list').addEventListener('click', event => {
    const button = event.target.closest('[data-vehicle]');
    if (!button) return;
    choose(button.dataset.vehicle);
    $('vehicle-list').querySelector(`[data-vehicle="${state.selected}"]`)?.focus({preventScroll:true});
  });
  document.querySelectorAll('[data-filter]').forEach(button => button.addEventListener('click', () => {state.filter=button.dataset.filter;renderList();}));
  $('scope-select').addEventListener('change', event => {
    state.scope = event.target.value;
    selectAvailable();
    state.segmentIndex = null;
    render();
    if (map && scopedVehicles().length) map.fitBounds(L.latLngBounds(scopedVehicles().map(v => v.position)),{padding:[55,55],maxZoom:12,animate:false});
  });
  $('threshold-select').addEventListener('change', event => { state.threshold = Number(event.target.value) * 60; render(); });
  document.querySelectorAll('[data-map-style]').forEach(button => button.addEventListener('click', () => setMapStyle(button.dataset.mapStyle)));
  document.querySelectorAll('[data-overlay]').forEach(button => button.addEventListener('click', () => {
    state[button.dataset.overlay] = !state[button.dataset.overlay];
    button.setAttribute('aria-pressed', String(state[button.dataset.overlay]));
    drawMap();
  }));
  $('segment-list').addEventListener('click', event => {
    const button = event.target.closest('[data-segment-index]');
    if (!button) return;
    state.segmentIndex = Number(button.dataset.segmentIndex);
    state.stops = true;
    document.querySelector('[data-overlay="stops"]').setAttribute('aria-pressed', 'true');
    renderSegments(); drawMap();
    const v = frame().vehicles.find(item => item.id === state.selected);
    const segment = routeSegments(v)[state.segmentIndex];
    if (map && segment) map.fitBounds(L.latLngBounds([segment.from.position, segment.to.position]), {padding:[90,90],maxZoom:15,animate:false});
    document.querySelector('.map-stage').scrollIntoView({behavior:'smooth',block:'center'});
  });
  $('incident-banner').addEventListener('click', () => {
    const v = frame().vehicles.find(item => item.id === state.selected);
    const segment = routeSegments(v).find(item => item.issue);
    if (!segment) return;
    state.segmentIndex = segment.index;
    renderSegments(); drawMap();
    map?.fitBounds(L.latLngBounds([segment.from.position, segment.to.position]), {padding:[90,90],maxZoom:15,animate:false});
  });
  $('search').addEventListener('input', event => { state.query = event.target.value; renderList(); });
  $('overview-button').addEventListener('click', () => {
    if (map && scopedVehicles().length) map.fitBounds(L.latLngBounds(scopedVehicles().map(v => v.position)),{padding:[55,55],maxZoom:12,animate:false});
  });
  $('about-button').addEventListener('click', () => $('about-dialog').showModal());
  $('sidebar-toggle').addEventListener('click', () => setSidebarOpen(false));
  $('sidebar-open').addEventListener('click', () => setSidebarOpen(true));
  $('map-cards-toggle').addEventListener('click', () => setCleanMap(!mapStage.classList.contains('clean-map')));
  $('map-expand-toggle').addEventListener('click', () => setMapExpanded(!workspace.classList.contains('map-expanded')));
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && workspace.classList.contains('map-expanded')) setMapExpanded(false);
  });
  $('close-about').addEventListener('click', () => $('about-dialog').close());
  $('understood').addEventListener('click', () => $('about-dialog').close());
})();
