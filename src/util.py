import pickle

def load_object(filename):
    try:
        with open(filename, "rb") as f:
            obj = pickle.load(f)

        return obj
    except Exception as e:
        print(e)
        return False