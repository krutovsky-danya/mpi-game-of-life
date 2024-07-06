#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "mpi.h"
#include <windows.h>

using namespace std;

const int DEAD_CELL = '0';
const int LIVE_CELL = '1';

vector<vector<int>> grid, nextGrid;


int countNeighbours(int h, int w, int x, int y)
{
	int countNeighbours = 0;
	for (int i = x - 1; i <= x + 1; i++)
	{
		if (0 > i || i >= h) {
			continue;
		}
		for (int j = y - 1; j <= y + 1; j++)
		{
			if (0 > j || j >= w)
				continue;
			if (i == x && j == y)
				continue;
			if (grid[i][j] == LIVE_CELL)
				countNeighbours++;
		}
	}
	return countNeighbours;
}

void updateGrid()
{
	size_t h = grid.size();
	size_t w = grid[0].size();
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			int countOfAliveNeighbours = countNeighbours(h, w, i, j);
			if (countOfAliveNeighbours == 3) {
				nextGrid[i][j] = LIVE_CELL;
			}
			if (countOfAliveNeighbours == 2 && grid[i][j] == LIVE_CELL)
				nextGrid[i][j] = LIVE_CELL;
			else
				nextGrid[i][j] = DEAD_CELL;
		}
	}
	swap(grid, nextGrid);
}

void readGridFromFile(int total_rows, int num_columns)
{
	MPI_Comm comm = MPI_COMM_WORLD;
	int rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	const char* file_name = "data.txt";
	MPI_File file;

	// Open the file using MPI I/O
	MPI_File_open(comm, file_name, MPI_MODE_RDONLY, MPI_INFO_NULL, &file);

	// Define file offset and chunk sizes for each process
	MPI_Offset chunk_size = total_rows / size;
	MPI_Offset start_row = rank * chunk_size;
	if (rank != 0) {
		start_row--; // take row from upper neighbour
		chunk_size++;
	}
	if (rank == size - 1) {
		chunk_size += total_rows % size; // take untaken tail
	}
	else {
		chunk_size += 1; // take row from lower neighbour;
	}

	// Read only the portion of the file assigned to this process
	std::vector<char> chunk_data(chunk_size * num_columns);
	MPI_File_read_at(file, start_row * num_columns * sizeof(char), chunk_data.data(), chunk_size * num_columns, MPI_CHAR, MPI_STATUS_IGNORE);

	// Process the data (e.g., calculate sum, average, etc.)
	grid = vector<vector<int>>(chunk_size, vector<int>());
	nextGrid = vector<vector<int>>(chunk_size, vector<int>(num_columns - 1, 0));

	int y = 0;
	for (int i = 0; i < chunk_data.size(); i++)
	{
		if (chunk_data[i] == '\n') {
			y++;
			continue;
		}
		grid[y].push_back(chunk_data[i]);
	}

	MPI_File_close(&file);
}

void exchangeGridDataWithUpperNeighbour(int rank, int iter) {
	if (rank == 0) {
		return;
	}

	int dest = rank - 1;
	vector<int> sendbuf = grid[1];
	vector<int> recvbuf = grid[0];

	MPI_Sendrecv(
		sendbuf.data(), sendbuf.size(), MPI_INT, dest, iter,
		recvbuf.data(), recvbuf.size(), MPI_INT, dest, iter + 1,
		MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void exchangeGridDataWithLowerNeighbour(int rank, int size, int iter) {
	int dest = rank + 1;
	if (dest == size) {
		return;
	}

	size_t h = grid.size();
	vector<int> sendbuf = grid[h - 2];
	vector<int> recvbuf = grid[h - 1];

	MPI_Sendrecv(
		sendbuf.data(), sendbuf.size(), MPI_INT, dest, iter + 1,
		recvbuf.data(), recvbuf.size(), MPI_INT, dest, iter,
		MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void exchangeGridData(int rank, int size, int iter) {
	if (rank % 2 == 0) {
		exchangeGridDataWithUpperNeighbour(rank, iter);
		exchangeGridDataWithLowerNeighbour(rank, size, iter);
	}
	else
	{
		exchangeGridDataWithLowerNeighbour(rank, size, iter);
		exchangeGridDataWithUpperNeighbour(rank, iter);
	}
}

void writeDataToFile(int rank, int size, int rows, int columns) {
	vector<char> data;
	size_t begin = 0, end = grid.size();
	if (rank != 0) {
		begin++;
	}
	if (rank != size - 1) {
		end--;
	}

	for (size_t y = begin; y < end; y++)
	{
		for (size_t x = 0; x < columns; x++)
		{
			data.push_back(grid[y][x]);
		}
		data.push_back('\n');
	}

	std::ofstream outputFile("output.txt", std::ios::app); // Open the file in append mode

	if (outputFile.is_open()) {
		MPI_File fileHandle;
		MPI_File_open(MPI_COMM_WORLD, "output.txt", MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fileHandle);

		MPI_Offset chunk_size = rows / size;
		MPI_Offset start_row = rank * chunk_size;
		MPI_Offset offset = start_row * (columns + 1) * sizeof(char); // Calculate the offset for each process based on rank
		MPI_File_write_at_all(fileHandle, offset, data.data(), data.size(), MPI_CHAR, MPI_STATUS_IGNORE);

		MPI_File_close(&fileHandle);

		std::cout << "Process " << rank << " wrote data to the file." << std::endl;

		outputFile.close(); // Close the local stream

		MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processes before proceeding
	}
	else {
		std::cerr << "Error opening the file for writing." << std::endl;
	}
}

int main(int argc, char** argv)
{
	int size, rank;
	int h, w, epochs;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	// printf("rank: %d \t size: %d \n", rank, size);
	double Time_work = MPI_Wtime();

	ifstream grid_size_data("grid_size_data.txt");

	if (grid_size_data >> h >> w >> epochs) {
	}
	else {
		std::cerr << "Error reading integers from file." << std::endl;
	}

	grid_size_data.close();

	readGridFromFile(h, w + 1);

	MPI_Barrier(MPI_COMM_WORLD);

	for (int i = 0; i < epochs; i++) {
		updateGrid();

		exchangeGridData(rank, size, i);

		MPI_Barrier(MPI_COMM_WORLD);
	}

	/*for (int y = 0; y < grid.size(); y++)
	{
		for (int x = 0; x < w; x++) {
			cout << (char)grid[y][x];
		}
		cout << endl;
	}*/

	writeDataToFile(rank, size, h, w);

	Time_work = MPI_Wtime() - Time_work;
	if (rank == 0)
	{
		cout << "Total time = " << Time_work << endl;
	}
	MPI_Finalize();
	return 0;
}