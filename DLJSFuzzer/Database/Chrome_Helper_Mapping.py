from Database.my_mysql import Session


def insert_chrome_helper_tensor(type, total):
    session = Session()
    sql = f"insert into chrome_helper values(\'TRUE\', \'{type}\', {total}, NULL)"
    cursor = session.execute(sql)
    session.commit()
    session.close()

def insert_chrome_helper_error(type, err_mag):
    session = Session()
    sql = f"insert into chrome_helper values(\'TRUE\', \'{type}\', 0, \'{err_mag}\')"
    cursor = session.execute(sql)
    session.commit()
    session.close()


def get_ready_then_fetch():
    session = Session()
    sql = "SELECT * FROM chrome_helper WHERE ready=\'TRUE\'"
    cursor = session.execute(sql)
    result = cursor.fetchall()
    session.close()
    if len(result) <= 0:
        return None
    ret = [[element[1], element[2], element[3]] for element in result]
    return ret[0]



def check_finish_and_get_last_id():
    session = Session()
    sql = "SELECT * FROM chrome_helper WHERE finish=\'TRUE\'"
    cursor = session.execute(sql)
    result = cursor.fetchall()
    session.close()
    if len(result) <= 0:
        return -1
    return result[0][2]

def clear_chrome_finish():
    session = Session()
    sql = "truncate table chrome_helper"
    cursor = session.execute(sql)
    session.commit()
    session.close()


# if get_chrome_finish() is None:
#     print("Fail")
# else:
#     print(get_chrome_finish())
# clear_chrome_finish()
# print(get_chrome_finish())
# insert_chrome_helper_tensor("tensor", 10)


