import pymysql
import os

class DBLoader(object):
    def __init__(self, 
                 host='', 
                 port=None, 
                 user='', 
                 password='', 
                 db=''):

        self.conn = pymysql.connect(host=host, port=port, user=user, password=password, db=db, charset='utf8')
        self.curs = self.conn.cursor()
        
    def sql_request(self, sql=""):
        
        self.curs.execute(sql)
        return self.curs
    
    def get_filenames_dict(self, sql):
        filenames = []
        self.curs.execute(sql)
        for row in self.curs:
            filenames.append(row[0].encode())
        return dict(zip(filenames, range(len(filenames))))
    
    def get_filenames_from_list(self, cursor, dirpath = "/data/files/oldvato/Win32 EXE", filename_idx = 0, label_idx = 1, base_idx = 2):
        
        filenames = []
        labels = []
        for row in cursor:
            md5 = row[2].encode()
            if len(md5) > 30:
                if row[1] != None:
                    if row[1].encode() != "CLEAN":
                        base = row[1].encode()
                        label = row[1].encode()
                        labels.append(label)
                        filenames.append(os.path.join(dirpath, base, md5))
        
        return filenames, labels
    
    def get_filenames_from_list_temp(self, cursor, dirpath = "/data/files/oldvato/Win32 EXE", filename_idx = 0, label_idx = 1, base_idx = 1):
        filenames = []
        labels = []
        for row in cursor:
            md5 = row[filename_idx]
            if row[label_idx] != None:
                #if row[label_idx].encode() != "CLEAN":
                label = row[label_idx]
                #base = row[base_idx]
                labels.append(label)
                filenames.append(md5)
                #filenames.append(os.path.join(dirpath, base, md5))
                
            
        return filenames, labels
    
    def close_conn(self):
        self.conn.close()