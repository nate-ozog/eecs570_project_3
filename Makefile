CC        = g++
CUDA_INC  = -I/usr/local/cuda/include/
BT_SRC = bt.cpp

bt: $(BT_SRC)
	$(CC) $(CFLAGS) $(BT_SRC) -o $@.o $(CUDA_INC)

clean:
	rm -rf *.o