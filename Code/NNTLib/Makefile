CC = g++
CPP_FILES = $(wildcard *.cpp)
OBJ_FILES = $(notdir $(CPP_FILES:.cpp=.o))
LD_FLAGS =  -fopenmp
CC_FLAGS =-Wall -g -O3 -std=c++11 -fopenmp

all: main


debug: CC_FLAGS+= -DDEBUG 
debug: $(OBJ_FILES)
	$(CC) $(LD_FLAGS) -o $@ $^

main: $(OBJ_FILES)
	$(CC) $(LD_FLAGS) -o $@ $^

%.o: %.cpp
	$(CC) $(CC_FLAGS) -c -o $@ $< $(LDFLAGS)

$(TARGET): $(OBJ_FILES)
	$(CC) $(OBJ_FILES) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(OBJ_FILES) $(TARGET) core main debug
