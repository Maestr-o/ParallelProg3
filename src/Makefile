CC=gcc
CFLAGS=-Wall -Wextra
BUILD_SEQ=../build/seq
BUILD_PAR=../build/par

all: seq parallel clean

seq: seq.o
	@$(CC) $(CFLAGS) -fopenmp seq.o -o $(BUILD_SEQ)

seq.o: seq.c
	@$(CC) $(CFLAGS) -std=c99 -c seq.c -o seq.o

parallel: parallel.o
	@$(CC) $(CFLAGS) -I /usr/local/cuda/include/ -L /usr/local/cuda/lib64/ -lOpenCL -fopenmp parallel.o -o $(BUILD_PAR)

parallel.o: parallel.c
	@$(CC) $(CFLAGS) -std=c99 -I /usr/local/cuda/include/ -L /usr/local/cuda/lib64/ -c parallel.c -o parallel.o

clean:
	@rm -f *.o

rebuild:
	clean all