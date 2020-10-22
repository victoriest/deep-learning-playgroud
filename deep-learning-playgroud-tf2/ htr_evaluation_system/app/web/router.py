from flask import request, jsonify, render_template, redirect, url_for

from . import router_blue_print
from ..model.models import *

common_api_headers = {
    'content-type': 'application/json'
}


@router_blue_print.route('/')
def index():
    page = 1
    limit = 20
    result = THtrDatum.query.filter(THtrDatum.pred_result == 'G').filter_by(is_verificated=0).limit(limit).offset(
        (page - 1) * limit).all()
    j = []
    for item in result:
        i = item.serialize()
        # print(item, i)
        j.append(i)

    return render_template('index.html', results=j)


# 分页查询未校正的选项图片列表
@router_blue_print.route('/regulation_list', methods={"GET"})
def get_regulation_list():
    page = int(request.args.get("page"))
    limit = int(request.args.get("limit"))
    result = THtrDatum.query.filter_by(is_verificated=0).limit(limit).offset((page - 1) * limit).all()
    j = []
    for item in result:
        i = item.serialize()
        # print(item, i)
        j.append(i)
    return jsonify(j), 200, common_api_headers


# 提交校正后的图片结果
@router_blue_print.route('/regulate', methods={"POST"})
def regulate():
    params = request.form
    for p, v in params.items():
        if v is None or v == '':
            print('delete', p, str(v))
            result = THtrDatum.query.filter_by(id=int(p)).first()
            db.session.delete(result)
            db.session.commit()
        else:
            print('update', p, str(v))
            THtrDatum.query.filter_by(id=int(p)).update(dict(real_result=str(v).upper(), is_verificated=1))
            db.session.commit()

    return redirect(url_for('web.index'))


def __query_failed_by_model_id(model_id):
    data = []
    for r, d in db.session.query(THtrModelPredResult, THtrDatum). \
            filter(THtrDatum.id == THtrModelPredResult.t_htr_data_id, THtrModelPredResult.t_htr_model_id == model_id). \
            filter(THtrModelPredResult.pred_result != THtrModelPredResult.real_result).all():
        i = r.serialize()
        j = d.serialize()
        data.append(
            {'id': j['id'], 'real_result': i['real_result'], 'pred_result': i['pred_result'], 'img_url': j['img_url']})
    return data


@router_blue_print.route('/failed-review', methods={"GET", "POST"})
def failed_review():
    results = {"model1": 2, "model2": 2, "rows": []}
    if request.method == 'GET':
        model_id = request.args.get("model-id")
        if model_id is None:
            model_id = 2
        else:
            model_id = int(model_id)
        data = __query_failed_by_model_id(model_id)
        rows = []
        for i in data:
            r = {"id": i["id"], "real_result": i['real_result'], "data1": i, "data2": None}
            rows.append(r)
        results['rows'] = rows
        results['model1'] = model_id
        results['model2'] = model_id

        return render_template('failed-review.html', results=results)
    else:
        model_id_1 = int(request.form['model1'])
        model_id_2 = int(request.form['model2'])

        data1 = __query_failed_by_model_id(model_id_1)
        data2 = __query_failed_by_model_id(model_id_2)

        ids_v = {}
        for i in data1:
            r = {"id": i["id"], "real_result": i['real_result'], "data1": i, "data2": None}
            ids_v[i["id"]] = r
        for i in data2:
            if i["id"] in ids_v:
                ids_v[i["id"]]['data2'] = i
            else:
                r = {"id": i["id"], "real_result": i['real_result'], "data1": None, "data2": i}
                ids_v[i["id"]] = r

        results['rows'] = ids_v.values()
        results['model1'] = model_id_1
        results['model2'] = model_id_2

        return render_template('failed-review.html', results=results)
