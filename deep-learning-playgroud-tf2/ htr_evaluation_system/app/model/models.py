# coding: utf-8
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


# https://greyli.com/generate-flask-sqlalchemy-model-class-for-exist-database/
# sqlacodegen mysql+pymysql://root:root@192.168.31.227:3306/htr_db > models.py
# flask-sqlacodegen --flask --outfile models.py mysql+pymysql://root:root@192.168.31.227:3306/htr_db
class Serializable:
    def serialize(self):
        """ Return object data in easily serializeable format"""
        result = {}
        for name, value in vars(self).items():
            if callable(name) or name.startswith("__") or name.startswith("_"):
                continue
            result[name] = value
        return result

class THtrDatum(db.Model, Serializable):
    __tablename__ = 't_htr_data'

    id = db.Column(db.Integer, primary_key=True)
    img_url = db.Column(db.String(512), nullable=False)
    img_cache_path = db.Column(db.String(512))
    pred_result = db.Column(db.String(16), nullable=False)
    real_result = db.Column(db.String(16))
    data_time = db.Column(db.DateTime, nullable=False, server_default=db.FetchedValue())
    is_verificated = db.Column(db.Integer, nullable=False, server_default=db.FetchedValue())


class THtrModel(db.Model, Serializable):
    __tablename__ = 't_htr_model'

    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(128))
    model_type = db.Column(db.String(45))


class THtrModelPredResult(db.Model, Serializable):
    __tablename__ = 't_htr_model_pred_result'

    id = db.Column(db.Integer, primary_key=True)
    t_htr_data_id = db.Column(db.ForeignKey('t_htr_data.id'), nullable=False, index=True)
    t_htr_model_id = db.Column(db.ForeignKey('t_htr_model.id'), nullable=False, index=True)
    pred_result = db.Column(db.String(16), nullable=False)
    real_result = db.Column(db.String(16), nullable=False)

    t_htr_data = db.relationship('THtrDatum', primaryjoin='THtrModelPredResult.t_htr_data_id == THtrDatum.id',
                                 backref='t_htr_model_pred_results')
    t_htr_model = db.relationship('THtrModel', primaryjoin='THtrModelPredResult.t_htr_model_id == THtrModel.id',
                                  backref='t_htr_model_pred_results')
