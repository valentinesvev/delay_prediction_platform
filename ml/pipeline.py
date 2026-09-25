"""Обработка входного события и подготовка признаков без операций с БД."""


def generate_features(event):
    return {'state': event['state'], 'delay': event['delay'], 'data_as_of': event['event_at']}
