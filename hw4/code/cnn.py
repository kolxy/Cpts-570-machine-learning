import util

def main():
    x, y = util.load_mnist()
    print(x.shape)
    return

if __name__ == "__main__":
    main()