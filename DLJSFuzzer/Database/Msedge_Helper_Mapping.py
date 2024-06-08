from Database.my_mysql import Session


def add_msedge_finish(finish, type, id, content):
    session = Session()
    sql = f"insert into msedge_helper values(\'{finish}\', \'{type}\', \'{id}\', \'{content}\')"
    cursor = session.execute(sql)
    session.commit()
    session.close()

def get_msedge_finish():
    session = Session()
    sql = "SELECT * FROM msedge_helper WHERE finish = \'True\'"
    cursor = session.execute(sql)
    result = cursor.fetchall()
    session.close()
    if len(result) <= 0:
        return None
    return result[0][1], result[0][2], result[0][3]

def clear_msedge_finish():
    session = Session()
    sql = "truncate table msedge_helper"
    cursor = session.execute(sql)
    session.commit()
    session.close()


# add_msedge_finish("True", "tensor", "20.001,30.989,100.899")
# if get_msedge_finish() is None:
#     print("Fail")
# else:
#     print(get_msedge_finish())
# clear_msedge_finish()


