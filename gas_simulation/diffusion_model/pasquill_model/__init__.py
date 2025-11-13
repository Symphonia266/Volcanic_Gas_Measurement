import os
import re

# パッケージの絶対パスを取得
package_path = ''.join(re.split(r'(\\)', __file__)[:-1])
