#include <iostream>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <mpi.h>

using namespace std;
using namespace cv;
using namespace chrono;

int main() {
	int rank, size;
	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	string input_filename, output_filename;
	Mat image;
	if (rank == 0) {
		cout << "\nWelcome to Parallel Image Processing with MPI\n\n";
		cout << "Please enter the filename of the input image (e.g., input.jpg): ";
		cin >> input_filename;

		image = imread(input_filename, IMREAD_COLOR);
		if (image.empty()) {
			cerr << "Failed to load image\n";
			MPI_Finalize();
			return 1;
		}
	}

	int rows = 0, cols = 0;
	if (rank == 0) {
		rows = image.rows;
		cols = image.cols;
	}
	MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

	int local_rows = rows / size;
	int local_image_size = local_rows * cols;
	Mat local_image(local_rows, cols, CV_8UC3);
	MPI_Scatter(image.data, local_image_size * 3, MPI_BYTE, local_image.data, local_image_size * 3, MPI_BYTE, 0, MPI_COMM_WORLD);

	const string image_operations[] = {
		"Gaussian Blur", "Edge Detection", "Image Scaling",
		"Histogram Equalization", "Color Space Conversion", "Global Thresholding",
		"Local Thresholding", "Median"
	};
	int choice = 0, operations_number = 8;
	do {
		if (rank == 0) {
			cout << "\nPlease choose an image processing operation:\n";
			for (int i = 0; i < operations_number; ++i) {
				cout << "\n0" << (i + 1) << "- " << image_operations[i];
			}
			cout << "\n\nEnter your choice (1-" << operations_number << "): ";
			cin >> choice;
			cout << "\nYou have selected " << image_operations[choice - 1] << ".\n";
			cout << "Please enter the filename for the output image (e.g., output.jpg): ";
			cin >> output_filename;
		}

		MPI_Bcast(&choice, 1, MPI_INT, 0, MPI_COMM_WORLD);
		auto start_time = high_resolution_clock::now();
		Mat processed_image;

		switch (choice) {
			case 1:
			{
				int blur_radius;
				if (rank == 0) {
					cout << "Please enter the blur radius (odd number: 3 or 5 or 7): ";
					cin >> blur_radius;
					cout << "Processing image " << input_filename << " with " << image_operations[choice - 1] << "...\n";
				}

				MPI_Bcast(&blur_radius, 1, MPI_INT, 0, MPI_COMM_WORLD);

				Mat blurred_image;
				GaussianBlur(local_image, blurred_image, Size(blur_radius, blur_radius), 0, 0);

				if (rank == 0) {
					processed_image = Mat(rows, cols, CV_8UC3);
				}

				MPI_Gather(blurred_image.data, local_image_size * 3, MPI_BYTE, processed_image.data, local_image_size * 3, MPI_BYTE, 0, MPI_COMM_WORLD);

				break;
			}
			case 2:
			{
				int threshold1, threshold2;
				if (rank == 0) {
					cout << "Please enter the lower threshold (e.g., 50): ";
					cin >> threshold1;
					cout << "Please enter the upper threshold (e.g., 150): ";
					cin >> threshold2;
					cout << "Processing image " << input_filename << " with " << image_operations[choice - 1] << "...\n";
				}

				MPI_Bcast(&threshold1, 1, MPI_INT, 0, MPI_COMM_WORLD);
				MPI_Bcast(&threshold2, 1, MPI_INT, 0, MPI_COMM_WORLD);

				Mat gray_image, edges;
				cvtColor(local_image, gray_image, COLOR_BGR2GRAY);
				Canny(gray_image, edges, threshold1, threshold2, 3);

				vector<int> recvcounts(size, local_image_size);
				vector<int> displs(size, 0);

				if (rank == 0) {
					processed_image = Mat::zeros(rows, cols, CV_8UC1);
					displs[0] = 0;
					for (int i = 1; i < size; i++) {
						displs[i] = displs[i - 1] + recvcounts[i - 1];
					}
				}

				MPI_Gatherv(edges.data, local_image_size * 1, MPI_UNSIGNED_CHAR, processed_image.data, recvcounts.data(), displs.data(), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

				break;
			}
			case 3: {
				double scale_factor;
				if (rank == 0) {
					cout << "Please enter the scaling factor (e.g., 0.5 for half-size, 2.0 for double-size): ";
					cin >> scale_factor;
				}

				MPI_Bcast(&scale_factor, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

				Mat scaled_image;
				resize(local_image, scaled_image, Size(), scale_factor, scale_factor);

				if (rank == 0) {
					processed_image = Mat(rows * scale_factor, cols * scale_factor, CV_8UC3);
				}

				MPI_Gather(scaled_image.data, local_image_size * 3 * scale_factor * scale_factor, MPI_BYTE, processed_image.data, local_image_size * 3 * scale_factor * scale_factor, MPI_BYTE, 0, MPI_COMM_WORLD);

				break;
			}
			case 4: {
				cvtColor(local_image, local_image, COLOR_BGR2GRAY);
				Mat equalized_image;
				equalizeHist(local_image, equalized_image);

				vector<int> recvcounts(size, local_image_size);
				vector<int> displs(size, 0);

				if (rank == 0) {
					processed_image = Mat::zeros(rows, cols, CV_8UC1);
					displs[0] = 0;
					for (int i = 1; i < size; i++) {
						displs[i] = displs[i - 1] + recvcounts[i - 1];
					}
				}

				MPI_Gatherv(equalized_image.data, local_image_size * 1, MPI_UNSIGNED_CHAR, processed_image.data, recvcounts.data(), displs.data(), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

				break;
			}
			case 5: {
				int conversion_code;
				if (rank == 0) {
					cout << "Please enter the conversion code:\n";
					cout << "0: RGB to Grayscale\n";
					cout << "1: RGB to HSV\n";
					cout << "2: RGB to LAB\n";
					cout << "Enter the conversion code (0, 1, or 2): ";
					cin >> conversion_code;
				}

				MPI_Bcast(&conversion_code, 1, MPI_INT, 0, MPI_COMM_WORLD);

				Mat converted_image;
				switch (conversion_code) {
					case 0:
						cvtColor(local_image, converted_image, COLOR_BGR2GRAY);
						break;
					case 1:
						cvtColor(local_image, converted_image, COLOR_BGR2HSV);
						break;
					case 2:
						cvtColor(local_image, converted_image, COLOR_BGR2Lab);
						break;
					default:
						if (rank == 0) cout << "Invalid conversion code!\n";
				}

				if (rank == 0) {
					processed_image = Mat(rows, cols, converted_image.type());
				}

				MPI_Gather(converted_image.data, local_image_size * converted_image.channels(), MPI_BYTE, processed_image.data, local_image_size * converted_image.channels(), MPI_BYTE, 0, MPI_COMM_WORLD);

				break;
			}
			case 6: {
				int threshold_value;
				if (rank == 0) {
					cout << "Please enter the threshold value (e.g., 128): ";
					cin >> threshold_value;
				}

				MPI_Bcast(&threshold_value, 1, MPI_INT, 0, MPI_COMM_WORLD);

				if (local_image.channels() > 1) {
					cvtColor(local_image, local_image, COLOR_BGR2GRAY);
				}

				Mat global_thresholded;
				threshold(local_image, global_thresholded, threshold_value, 255, THRESH_BINARY);

				if (rank == 0) {
					processed_image = Mat(rows, cols, CV_8UC1);
				}

				MPI_Gather(global_thresholded.data, local_image_size, MPI_BYTE, processed_image.data, local_image_size, MPI_BYTE, 0, MPI_COMM_WORLD);

				break;
			}
			case 7: {
				int blockSize, offset;
				if (rank == 0) {
					cout << "Please enter the block size (odd number: 3 or 5 or 7): ";
					cin >> blockSize;
					cout << "Please enter the offset value (e.g., 10): ";
					cin >> offset;
				}

				MPI_Bcast(&blockSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
				MPI_Bcast(&offset, 1, MPI_INT, 0, MPI_COMM_WORLD);

				if (local_image.channels() > 1) {
					cvtColor(local_image, local_image, COLOR_BGR2GRAY);
				}

				Mat local_thresholded;
				adaptiveThreshold(local_image, local_thresholded, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, blockSize, offset);

				if (rank == 0) {
					processed_image = Mat(rows, cols, CV_8UC1);
				}

				MPI_Gather(local_thresholded.data, local_image_size, MPI_BYTE, processed_image.data, local_image_size, MPI_BYTE, 0, MPI_COMM_WORLD);

				break;
			}
			case 8: {
				int kernel_size;
				if (rank == 0) {
					cout << "Please enter the kernel size (odd number: 3 or 5 or 7): ";
					cin >> kernel_size;
				}

				MPI_Bcast(&kernel_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

				Mat filtered_image;
				medianBlur(local_image, filtered_image, kernel_size);

				if (rank == 0) {
					processed_image = Mat(rows, cols, CV_8UC3);
				}

				MPI_Gather(filtered_image.data, local_image_size * 3, MPI_BYTE, processed_image.data, local_image_size * 3, MPI_BYTE, 0, MPI_COMM_WORLD);

				break;
			}
			default: if (rank == 0) cout << "Invalid choice!\n"; break;
		}

		auto end_time = high_resolution_clock::now();
		auto duration = duration_cast<seconds>(end_time - start_time);

		if (rank == 0) {
			imwrite(output_filename, processed_image);
			cout << "\n" << image_operations[choice - 1] << " operation completed successfully in " << duration.count() << " seconds.\n\n";
			cout << "Image saved as " << output_filename << ".\n";
		}
	} while (choice);

	MPI_Finalize();
	return 0;
}