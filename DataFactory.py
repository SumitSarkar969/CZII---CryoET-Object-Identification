import zipfile
from abc import abstractmethod
import sys

class DataFactory:
    @abstractmethod
    def Ingest(self):
        pass

class ZipIngestor(DataFactory):
    def Ingest(self, data):
        print("Extracting...")
        with zipfile.ZipFile(data) as zip:
            zip.extractall('Extracted_Data')


class Processor:
    def getIngestor(self, data):
        if 'zip' in data.split('.'):
            return ZipIngestor()

    def processFile(self, data):
        Ingestor = self.getIngestor(data)
        Ingestor.Ingest(data)


if __name__ == '__main__':
    file =  sys.argv[1]
    Processor().processFile(file)
