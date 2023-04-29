import easyocr


def text_recognition(file_path):
    reader = easyocr.Reader(['ru'])
    result = reader.readtext(file_path, detail=0)

    return result


def main():
    file_path = 'images/' + input('Enter file path: ')
    print(text_recognition(file_path=file_path))


if __name__ == '__main__':
    main()
