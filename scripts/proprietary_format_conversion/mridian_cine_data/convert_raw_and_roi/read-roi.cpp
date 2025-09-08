#include <cstdio>
#include <vector>
#include <fstream>
#include <iostream>   // Include the header for file stream operations
#include <filesystem>  // Include the header for filesystem operations
#include <sstream>  // Include the header for stringstream operations

/*
The "Cine ID" is the index of the frame. "Points" is the number of points that represent the segmentation (i.e. the ROI - region of interest).
The coordinates of the points are then listed in (X, Y) format. The point (0.0, 0.0) corresponds to the upper left corner of the frame and (1.0, 1.0) - to the lower right corner.
*/

struct ROI {
    enum ContourMode {
        CRUDE,
        SOFT,
    };

    struct Vec2 { float x, y; };

    int cineId;

    bool needSoften;

    ContourMode contourMode;

    std::vector<Vec2> points;
};

bool read_from_file(const std::string & fname, std::vector<ROI> & contours) {
    contours.clear();

    std::ifstream fin(fname, std::ios::binary);
    if (!fin.good() || !fin.is_open()) {
        return false;
    }

    int nContours = 0;
    fin.read((char *)(&nContours), sizeof(nContours));

    contours.resize(nContours);
    for (int i = 0; i < nContours; ++i) {
        int cid = -1; fin.read((char *)(&cid), sizeof(cid));

        ROI curROI;

        curROI.cineId = cid;

        fin.read((char *)(&curROI.needSoften),  sizeof(curROI.needSoften));
        fin.read((char *)(&curROI.contourMode), sizeof(curROI.contourMode));

        int nPoints = -1; fin.read((char *)(&nPoints), sizeof(nPoints));
        for (int j = 0; j < nPoints; ++j) {
            ROI::Vec2 p;
            fin.read((char *)(&p.x), sizeof(p.x));
            fin.read((char *)(&p.y), sizeof(p.y));

            curROI.points.push_back(p);
        }

        contours[i] = std::move(curROI);
    }

    fin.close();
    return true;
}

//Save the coordinates of the contours to a txt file
int main(const int argc, const char * argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <contours.bin>\n", argv[0]);
        return 1;
    }

    std::vector<ROI> contours;

    if (!read_from_file(argv[1], contours)) {
        fprintf(stderr, "Failed to read contours from file: %s\n", argv[1]);
        return 1;
    }

    // Get the path of the input file
    std::filesystem::path inputFilePath(argv[1]);

    // Create a subdirectory named "converted_contours"
    std::filesystem::create_directory(inputFilePath.parent_path() / "converted_contours");

    for (int j = 0; j < (int)contours.size(); ++j) {
        const ROI & roi = contours[j];

        // Construct the output file path in the "converted_contours" subdirectory
        std::ostringstream outputFileName;
        outputFileName << "contours_frame_" << std::setw(4) << std::setfill('0') << roi.cineId << ".txt";
        std::filesystem::path outputFilePath = inputFilePath.parent_path() / "converted_contours" / outputFileName.str();

        // Open the output file for writing
        std::ofstream outFile(outputFilePath);

        if (!outFile.is_open()) {
            fprintf(stderr, "Failed to open output file for writing.\n");
            return 1;
        }

        // Write the points to the text file
        outFile << "Cine ID: " << roi.cineId << "\n";
        outFile << "Need soften: " << (roi.needSoften ? "true" : "false") << "\n";
        outFile << "Contour mode: " << (roi.contourMode == ROI::CRUDE ? "crude" : "soft") << "\n";
        outFile << "Points: " << (int)roi.points.size() << "\n";
        for (int k = 0; k < (int)roi.points.size(); ++k) {
            const ROI::Vec2 & p = roi.points[k];
            outFile << p.x << ", " << p.y << "\n";
        }

        // Close the output file
        outFile.close();
    }

    return 0;
}
