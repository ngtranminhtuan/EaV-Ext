import configparser

def getConfig(path='config.ini'):
    config = configparser.ConfigParser(comment_prefixes='/', allow_no_value=True)
    config.path = path
    config.optionxform = str
    config.sections()
    config.read(path)
    return config

def setConfig(config, key, value):
    config.set("DEFAULT", key, value)
    with open(config.path, "w") as f:
        config.write(f)