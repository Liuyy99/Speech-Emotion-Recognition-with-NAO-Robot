import sys

def hello():
    print "Hello World!"

if __name__ == '__main__':
    print sys.argv[0]
    print sys.argv[1]
    hello()