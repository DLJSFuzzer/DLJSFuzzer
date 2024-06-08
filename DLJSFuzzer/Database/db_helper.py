from Database.my_mysql import Session

def clear_chrome_helper():
    session = Session()
    sql = "truncate table chrome_helper"
    cursor = session.execute(sql)
    session.commit()
    session.close()


def clear_msedge_helper():
    session = Session()
    sql = "truncate table msedge_helper"
    cursor = session.execute(sql)
    session.commit()
    session.close()


def clear_input_helper():
    session = Session()
    sql = "truncate table input_helper"
    cursor = session.execute(sql)
    session.commit()
    session.close()

def clear_db():
    clear_input_helper()
    clear_chrome_helper()
    clear_msedge_helper()

# clear_db()
