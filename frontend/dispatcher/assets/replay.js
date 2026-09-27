/* Minimal controls for the shared historical clock; no client-side GPS replay. */
(() => {
  'use strict';
  const panel = document.getElementById('replay-controls');
  const form = document.getElementById('replay-form');
  const input = document.getElementById('replay-at');
  const beginning = document.getElementById('replay-beginning');
  const status = document.getElementById('replay-status');
  let session, pending = false, timer;
  function disable(value) {
    form.querySelectorAll('button').forEach(button => { button.disabled = Boolean(value); });
  }
  async function refresh() {
    clearTimeout(timer);
    try {
      const response = await fetch('/replay/status', {cache:'no-store', signal:AbortSignal.timeout(10000)});
      if (!response.ok) throw new Error('Не удалось получить состояние воспроизведения');
      const data = await response.json();
      panel.hidden = !data.enabled;
      if (!data.enabled) return;
      if ((session && data.session && data.session !== session) || (typeof pending === 'string' && data.request_id === pending)) {
        location.reload(); // Discard every response and map layer from the old run.
        return;
      }
      session = data.session || session;
      input.min = data.first.slice(0, 19);
      input.max = data.last.slice(0, 19);
      if (!input.value) input.value = (data.current || data.first).slice(0, 19);
      const labels = {initializing:'Подготавливаем расчёт', switching:'Переходим к выбранному времени',
        running:'Воспроизведение идёт', finished:'История закончилась. Можно начать сначала или выбрать время'};
      status.textContent = data.error ? 'Ошибка расчёта. Повторяем попытку; время пока не меняется.' :
        `${pending ? 'Команда отправлена. Ожидаем переключения' : labels[data.status] || data.status}${data.current ? ` · ${data.current.slice(0,19).replace('T',' ')} UTC` : ''}`;
      disable(pending || data.status === 'switching');
    } catch (error) {
      status.textContent = error.message;
    } finally {
      timer = setTimeout(refresh, 2000);
    }
  }
  async function start(at) {
    if (pending) return;
    pending = true;
    disable(true);
    status.textContent = 'Отправляем команду…';
    try {
      const response = await fetch('/replay/start', {method:'POST', headers:{'Content-Type':'application/json'},
        body:JSON.stringify({at}), signal:AbortSignal.timeout(10000)});
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Не удалось изменить время');
      pending = data.request_id;
      await refresh();
    } catch (error) {
      pending = false;
      disable(false);
      status.textContent = error.message;
    }
  }
  form.addEventListener('submit', event => {
    event.preventDefault();
    if (form.reportValidity()) start(`${input.value}Z`);
  });
  beginning.addEventListener('click', () => start(null));
  refresh();
})();
