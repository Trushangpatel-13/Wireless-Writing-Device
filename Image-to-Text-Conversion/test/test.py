import requests
import time
import base64
from json import dumps

image_path = "./test/2..png"
with open(image_path, "rb") as img_file:
    my_string = base64.b64encode(img_file.read())
print(my_string)
data = {'file':my_string}
#file = {'file':open('./test/2..png','rb')}
res = requests.post('https://canvas-image-to-text.herokuapp.com/ocr',data)
print(res.text)
