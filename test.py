import requests

url = 'http://127.0.0.1:8080'
myobj = {'ingredients': [
    '16 tablespoons (227g) unsalted butter, at room temperature, at least 65°F*',
    '2 cups (397g) granulated sugar',
    '2 teaspoons baking powder'
]}

x = requests.post(url, json = myobj)

print(x.text)