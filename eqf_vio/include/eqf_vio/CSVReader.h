#include <fstream>
#include <sstream>
#include <string>
#include <vector>

class CSVLine {
  protected:
    friend class CSVReader;
    std::vector<std::string> data;

    void readLine(std::istream& lineStream) {
        data.clear();
        std::string entry;
        while (std::getline(lineStream, entry, ',')) {
            data.emplace_back(entry);
        }
    }

  public:
    CSVLine() = default;
    CSVLine(std::istream& lineStream) { readLine(lineStream); }
    std::string operator[](const size_t& idx) const { return data[idx]; }
    size_t size() { return data.size(); }
};

class CSVReader {
  protected:
    std::istream* csvPtr;
    CSVLine csvLine;

  public:
    CSVReader() { csvPtr = NULL; }
    CSVReader(std::istream& f) {
        csvPtr = f.good() ? &f : NULL;
        readNextLine();
    }

    CSVReader begin() { return *this; }
    CSVReader end() { return CSVReader(); }
    CSVLine operator*() { return csvLine; }
    bool operator==(const CSVReader& other) {
        return (this->csvPtr == other.csvPtr) || ((this->csvPtr == NULL) && (other.csvPtr == NULL));
    }
    bool operator!=(const CSVReader& other) { return !(*this == other); }

    CSVReader& operator++() {
        if (csvPtr) {
            // Try to read the next line
            if (!readNextLine()) {
                csvPtr = NULL;
            }
        }
        return *this;
    }

    bool readNextLine() {
        std::string lineString;
        bool fileNotEmpty = (bool)std::getline(*csvPtr, lineString, '\n');
        if (fileNotEmpty) {
            std::stringstream lineStream(lineString);
            csvLine.readLine(lineStream);
        } else {
            csvLine = CSVLine();
        }
        return fileNotEmpty;
    }
};