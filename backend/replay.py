"""Replay commands write a small request file; only the coordinator writes DBs."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError

from storage import replay

router = APIRouter(prefix='/replay', tags=['historical replay'])


class StartRequest(BaseModel):
    at: str | None = None


@router.get('/status')
def status():
    if not replay.enabled():
        return {'enabled': False}
    try:
        state = replay.snapshot()
        first, last = replay.bounds() if not state else (state['first'], state['last'])
        request = replay.read_json('request.json')
        pending = bool(request and (not state or request['id'] != state['request_id']))
        return dict(enabled=True, first=replay.iso(first), last=replay.iso(last),
                    current=replay.iso(state['cursor']) if state else None,
                    session=state['session'] if state else None,
                    request_id=state['request_id'] if state else None,
                    status=('switching' if pending else state['status'] if state else 'initializing'),
                    error=state.get('error') if state else None)
    except (OSError, ValueError, SQLAlchemyError) as exc:
        raise HTTPException(503, 'Состояние исторического воспроизведения недоступно') from exc


@router.post('/start', status_code=202)
def start(body: StartRequest):
    if not replay.enabled():
        raise HTTPException(409, 'Управление доступно только в режиме исторического воспроизведения')
    try:
        request = replay.request_start(body.at)
        return dict(status='accepted', request_id=request['id'], start=replay.iso(request['start']))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except (OSError, SQLAlchemyError) as exc:
        raise HTTPException(503, 'Не удалось передать команду воспроизведения') from exc
