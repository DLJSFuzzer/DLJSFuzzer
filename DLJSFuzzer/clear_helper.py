from Method import util
from Database import db_helper
from DataStruct.globalConfig import GlobalConfig

util.clear_dir(f"{GlobalConfig.absolutePath}\\TFJS_output_storage\\Chrome_output_storage")
util.clear_dir(f"{GlobalConfig.absolutePath}\\TFJS_output_storage\\Edge_output_storage")
util.clear_dir(f"{GlobalConfig.absolutePath}\\TFJS_Model")
util.clear_dir(f"{GlobalConfig.absolutePath}\\Crush_logs")
db_helper.clear_db()

