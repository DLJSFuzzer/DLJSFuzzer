from Database.my_mysql import Session

def insert_input_helper(shape, content, url):
    session = Session()
    sql = f"insert into input_helper values(\'TRUE\', \'{shape}\', \'{content}\', \'{url}\')"
    cursor = session.execute(sql)
    session.commit()
    session.close()

def check_ready_then_fetch():
    session = Session()
    sql = "SELECT * FROM input_helper WHERE ready=\'TRUE\'"
    cursor = session.execute(sql)
    result = cursor.fetchall()
    session.close()
    if len(result) <= 0:
        return None
    ret = [[element[1], element[2], element[3]] for element in result]
    return ret

# For Random
def clear_input_helper():
    session = Session()
    sql = "truncate table input_helper"
    cursor = session.execute(sql)
    session.commit()
    session.close()


# For Dataset
def set_ready(modelUrl):
    session = Session()
    sql = f"update input_helper set ready=\'TRUE\',url=\'{modelUrl}\' where ready=\'FALSE\'"
    cursor = session.execute(sql)
    session.commit()
    session.close()


# For Dataset
def reset_ready():
    session = Session()
    sql = f"update input_helper set ready=\'FALSE\' where ready=\'TRUE\'"
    cursor = session.execute(sql)
    session.commit()
    session.close()


# insert_input_helper("1, 2, 2, 1", "1, 2, 3, 4", "model.json")
# reset_ready()
# set_ready('ttttt')
# clear_input_helper()

