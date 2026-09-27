// Offline control-flow tests; browser layout is checked separately when available.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('frontend/dispatcher/assets/replay.js', 'utf8');

function harness(enabled = true) {
  const elements = new Map();
  for (const id of ['replay-controls','replay-form','replay-at','replay-beginning','replay-status']) {
    elements.set(id, {hidden:true, value:'', disabled:false, textContent:'', listeners:{},
      addEventListener(type, fn) { this.listeners[type] = fn; }, reportValidity() {return true;}});
  }
  const submit = {disabled:false};
  elements.get('replay-form').querySelectorAll = () => [submit, elements.get('replay-beginning')];
  const state = {enabled, first:'2026-01-06T00:00:02+00:00', last:'2026-01-06T23:59:59+00:00',
    current:'2026-01-06T07:23:00+00:00', session:'old', request_id:'old-request', status:'running'};
  let timer, reloads = 0, failure = false;
  const posts = [];
  vm.runInNewContext(source, {document:{getElementById:id=>elements.get(id)}, AbortSignal,
    location:{reload() {reloads++;}}, clearTimeout(){}, setTimeout(fn) {timer=fn;},
    fetch: async (url, options) => {
      if (url === '/replay/status') return {ok:true, json:async()=>state};
      assert.equal(url, '/replay/start');
      posts.push(JSON.parse(options.body));
      return {ok:!failure, json:async()=>failure ? {detail:'Время вне истории'} : {request_id:'new-request'}};
    }});
  return {elements, submit, state, posts, tick:()=>new Promise(setImmediate), poll:()=>timer(),
    reloads:()=>reloads, fail:()=>{failure=true;}};
}

test('time selection is UTC and a new run discards the old page', async()=>{
  const h = harness(); await h.tick();
  assert.equal(h.elements.get('replay-controls').hidden, false);
  h.elements.get('replay-at').value = '2026-01-06T07:30';
  h.elements.get('replay-form').listeners.submit({preventDefault(){}});
  await h.tick();
  assert.deepEqual(h.posts, [{at:'2026-01-06T07:30Z'}]);
  assert.equal(h.submit.disabled, true);
  assert.equal(h.reloads(), 0);
  Object.assign(h.state, {session:'new', request_id:'new-request'});
  await h.poll();
  assert.equal(h.reloads(), 1);
});

test('beginning uses server history bounds and errors re-enable controls', async()=>{
  const h = harness(); await h.tick(); h.fail();
  h.elements.get('replay-beginning').listeners.click();
  await h.tick();
  assert.deepEqual(h.posts, [{at:null}]);
  assert.equal(h.submit.disabled, false);
  assert.match(h.elements.get('replay-status').textContent, /Время вне истории/);
});

test('controls stay hidden outside replay mode', async()=>{
  const h = harness(false); await h.tick();
  assert.equal(h.elements.get('replay-controls').hidden, true);
  assert.equal(h.posts.length, 0);
});
