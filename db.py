from sqlalchemy import create_engine

_connection = None

def get_connection():
    global _connection
    if not _connection:
        USER   = 'dementia'
        PASSWD = 'Dementia123!'
        DB     = 'masters_data'
        url = 'mysql://%s:%s@localhost/%s' % (USER, PASSWD, DB)
        engine = create_engine(url)
        _connection = engine.connect()
    return _connection