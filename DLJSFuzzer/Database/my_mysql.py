from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

# TODO 连接数据库，格式：mysql+mysqlconnector://{用户名}:{密码}@{ip地址}:{端口号}/tfjshelper
# ip地址和端口号一般都是localhost:3306
url = "mysql+mysqlconnector://root:yourpassword@localhost:3306/tfjshelper"
engine = create_engine(url)
Session = sessionmaker(bind=engine)