import requests

url = 'http://127.0.0.1:8080'
myobj = {'ingredients': [
    "1 clove of garlic, minced",
    "1 cup flour",
    "1 cup milk",
    "1 cup of sliced fresh spinach",
    "1 lb penne pasta",
    "1 tbps butter",
    "1 tbps flour",
    "1 tbps olive oil",
    "1 tbps salt",
    "1 tbps sugar",
    "1.5 cups of marinara sauce",
    "1/2 cup heavy cream",
    "1/2 cup shredded parmesan cheese",
    "1/4 or 1/2 tsp crushed red pepper flakes",
    "3-4 white mushroom, diced",
    "4 roma tomatoes diced",
    "Hello",
    "Ingredients",
    "Pee Pee",
    "salt to taste",
    "World"
]}

x = requests.post(url, json = myobj)

print(x.text)