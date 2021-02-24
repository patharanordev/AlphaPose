import requests
import pandas as pd
import os

class DataRecorder:
  def __init__(self):
    self.host = 'http://ec2-54-169-202-49.ap-southeast-1.compute.amazonaws.com'
    self.endpoint = '/api/ai-masks'
    self.url = "{}{}".format(self.host, self.endpoint)
    self.bearer = ''
    self.data = {}

  def set_bearer(self, bearer):
    self.bearer = bearer

  def update_server(self, date, time, week, status, camera_id, location, \
      periods, mask_total, wear_mask, wrong_mask, no_mask, note, district:int):
      
    payload='''{
      "date": "{}",
      "time": "{}",
      "week": {},
      "status": "{}",
      "camera_id": "{}",
      "location": "{}",
      "periods": "{}",
      "mask_total": {},
      "wear_mask": {},
      "wrong_mask": {},
      "no_mask": {},
      "note": "{}",
      "district": {}
    }
    '''.format(date, time, week, status, camera_id, location, \
      periods, mask_total, wear_mask, wrong_mask, no_mask, note, district)

    headers = {
      'Authorization': 'Bearer '.format(self.bearer),
      'Content-Type': 'application/json'
    }

    response = requests.request("POST", self.url, headers=headers, data=payload)
    print(response.text)

  def clear_data(self):
    self.data = {}

  def export_data(self, fpath, fname):
    pd.DataFrame(self.data).T.to_csv(os.path.join(fpath, '{}.csv'.format(fname)))
    self.clear_data()

  def append_data(self, id, flag):
    self.data[id] = { 'id':id, 'flag':flag }