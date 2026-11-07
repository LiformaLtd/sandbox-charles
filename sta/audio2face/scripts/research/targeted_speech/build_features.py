from common import *
from features import build
if __name__=='__main__':
 inputs()
 for arm in ['broad','targeted']:build(arm+'-train',OUT/'datasets'/arm,'train')
