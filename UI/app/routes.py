from flask import render_template, flash, redirect, url_for
from app import app
from app.forms import IndexForm, PredictionForm, AnalyseForm, AnalyseInitialForm, DistanceForm
from app.functionality import validate, predictStock, checkRange, graphPlot, DTWDistance
import ast, time
from wtforms import SelectField, FieldList
from wtforms.validators import DataRequired

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    form = IndexForm()
    if form.validate_on_submit():
        if (form.DTW.data):
            return redirect(url_for('analyse_initial'))
        elif (form.predict.data):
            return redirect(url_for('predict'))
        else:
            return redirect(url_for('DTW_initial'))
    return render_template('index.html', form=form)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    form = PredictionForm()
    if form.validate_on_submit():
        if validate(form.date.data) == 0:
            flash('Incorrect date entered. Please check!')
            return redirect(url_for('predict'))

        if checkRange(form.date.data) == 0:
            flash('Date out of range. Please enter between 1st February,2013 to 31st December,2019!')
            return redirect(url_for('predict'))

        if (form.company.data == 1):
            company = 'Toyata'
        else:
            company = 'GM'

        date = form.date.data
        return redirect(url_for('predict_render', company=company, date=form.date.data))
    return render_template('predict.html', form=form)


@app.route('/predict_render/<company>/<date>', methods=['GET', 'POST'])
def predict_render(company, date):
    result = predictStock(company, date)

    if(result == -1):
        flash('Date not present in the database. Please enter some other date!')
        return redirect(url_for('predict'))
    predicted = result[0]
    original = result[1]
    #flash(result)
    #return redirect(url_for('predict'))
    return render_template('predict_render.html', predicted=predicted, original=original, company=company)


@app.route('/analyse/<number>', methods=['GET', 'POST'])
def analyse(number):
    number = int(number)

    class LocalForm(AnalyseForm):pass
    LocalForm.company = FieldList(SelectField('Company: ', choices=[(1, 'Toyota'), (2, 'General Motors'), (3, 'Apple'), (4, 'Tesla'), (5, 'Google')], coerce=int, validators=[DataRequired()]), min_entries=number)
    form = LocalForm()

    if form.validate_on_submit():
        companies = list()

        for i in form.company.data:
            if ( i == 1):
                companies.append('Toyata')
            elif (i == 2):
                companies.append('GM')
            elif (i == 3):
                companies.append('Apple')
            elif (i == 4):
                companies.append('TSLA')
            else:
                companies.append('Google')

        return redirect(url_for('analyse_render', companies=companies))
    return render_template('analyse.html', form=form, number=number)


@app.route('/analyse_render/<companies>', methods=['GET', 'POST'])
def analyse_render(companies):
    companies = ast.literal_eval(companies)
    destination = '/home/saksham/Documents/microblog/app/static/plot' + str(time.time()) + '.png'
    graphPlot(companies, destination)
    return render_template('analyse_render.html', destination=destination[37:])



@app.route('/analyse_initial', methods=['GET', 'POST'])
def analyse_initial():
    form = AnalyseInitialForm()
    if form.validate_on_submit():
        if( int(form.number.data) < 1 or int(form.number.data) > 5):
            flash('Number entered should be between 1 and 5. Please Recheck')
            return redirect(url_for('analyse_initial'))

        return redirect(url_for('analyse', number=form.number.data))
    return render_template('analyse_initial.html', form=form)


@app.route('/DTW_initial', methods=['GET', 'POST'])
def DTW_initial():
    form = DistanceForm()
    if form.validate_on_submit():

        if (form.company1.data == 1):
            company1 = 'Toyata'
        elif (form.company1.data == 2):
            company1 = 'GM'
        elif (form.company1.data == 3):
            company1 = 'Apple'
        elif (form.company1.data == 4):
            company1 = 'TSLA'
        else:
            company1 = 'Google'

        if (form.company2.data == 1):
            company2 = 'Toyata'
        elif (form.company2.data == 2):
            company2 = 'GM'
        elif (form.company2.data == 3):
            company2 = 'Apple'
        elif (form.company2.data == 4):
            company2 = 'TSLA'
        else:
            company2 = 'Google'

        distance = DTWDistance(company1, company2)
        return render_template('DTW_render.html', distance=distance, company1=company1, company2=company2)
    return render_template('DTW_initial.html', form=form)

