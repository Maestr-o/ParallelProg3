CC=gcc
CFLAGS=-Wall -Wextra
BUILD=../build/prog
LDFLAGS=-fopenmp

all: build

build: main.o
	@$(CC) $(CFLAGS) $(LDFLAGS) main.o -o $(BUILD)

main.o: main.c
	@$(CC) $(CFLAGS) $(LDFLAGS) -c main.c -o main.o

clean:
	@rm -f *.o

rebuild:
	clean all