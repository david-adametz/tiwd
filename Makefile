CC=g++
CFLAGS=-O3 -c -Wall
LDFLAGS=-larmadillo -lboost_random
SOURCES=tiwd.cpp
OBJECTS=$(SOURCES:.cpp=.o)
	EXECUTABLE=tiwd

all: $(SOURCES) $(EXECUTABLE)
		
$(EXECUTABLE): $(OBJECTS) 
		$(CC) $(OBJECTS) -o $@ $(LDFLAGS) 

.cpp.o:
		$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f *.o tiwd
