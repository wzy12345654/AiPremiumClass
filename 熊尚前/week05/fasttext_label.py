import fasttext

model = fasttext.train_supervised('cooking.stackexchange.txt', epoch=10, lr=0.1, dim=100)

print(model.predict("Which baking dish is best to bake a banana bread ?"))
print(model.predict("Is it safe to eat food that was heated in plastic wrap to the point the plastic wrap flamed?"))