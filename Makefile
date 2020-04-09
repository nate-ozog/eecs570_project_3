CC        = g++
CUDA_INC  = -I/usr/local/cuda/include/
BT_SRC = bt.cpp
BT_HDR = nw_general.hpp

bt: $(BT_SRC) $(BT_HDR)
	$(CC) $(CFLAGS) $(BT_SRC) -o $@.o $(CUDA_INC)

clean:
	rm -rf *.o