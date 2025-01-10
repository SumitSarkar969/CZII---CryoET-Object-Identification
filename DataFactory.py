import zipfile
from abc import abstractmethod
import sys
from cryoet_data_portal import Client, Dataset

class DataFactory:
    @abstractmethod
    def Ingest(self, data):
        pass

class ZipIngestor(DataFactory):
    def Ingest(self, data):
        print("Extracting...")
        with zipfile.ZipFile(data[2]) as zip:
            zip.extractall('Extracted_Data')


class Extractor:
    def getIngestor(self, data):
        if 'zip' in data[2].split('.'):
            return ZipIngestor()

    def call(self, data):
        Ingestor = self.getIngestor(data)
        Ingestor.Ingest(data)


class CryoIngestor(DataFactory):
    def Ingest(self, data):
        client = Client()
        dataset = Dataset.get_by_id(client, int(data[2]))
        dataset.download_everything()

class Downloader:
    def get_ingestor(self, data):
        if data[1] == 'czii':
            return CryoIngestor()
    def call(self, data):
        ingestor = self.get_ingestor(data)
        ingestor.Ingest(data)


class Processor:
    def get_processor(self, data):
        process = data[0]
        if process not in ['-d','-e']:
            raise NotImplementedError

        if process == '-e':
            return Extractor()
        elif process == '-d':
            return Downloader()

    def process(self, data):
        process = self.get_processor(data)
        process.call(data)

if __name__ == '__main__':
    file =  sys.argv[1:]
    Processor().process(file)

