import platform


def main():
    l = [5,6,7]
    print(l[:10:20])

    computer = platform.node().upper()
    print(computer)

    computer = 'EDOUARD'
    if computer not in ['EDOUARD','SERVER']:
        print("Computer name not recognized")
        exit(1)

    return 0

if __name__ == '__main__':
    main()