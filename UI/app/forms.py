from flask_wtf import FlaskForm
from wtforms import SubmitField, SelectField, DateField, StringField, FieldList
from wtforms.validators import DataRequired


class IndexForm(FlaskForm):
    style1 = {'style': 'width:300px; font-size:25px; position:absolute; margin-left:30px;'}
    style2 = {'style': 'width:300px; font-size:25px; margin-left:350px; position:absolute; '}
    style3 = {'style': 'width:300px; font-size:25px; margin-left:675px;'}
    DTW = SubmitField('Stock Analysis', render_kw=style2)
    predict = SubmitField('Stock Prediction', render_kw=style3)
    distance = SubmitField('DTW Distance', render_kw=style1)


class PredictionForm(FlaskForm):
    dropdown_list = [(1, 'Toyata'), (2, 'General Motors')]
    company = SelectField('Company', choices=dropdown_list, coerce=int, validators=[DataRequired()])
    date = StringField('Prediction Date (in the format YYYY-MM-DD)', validators=[DataRequired()])
    submit = SubmitField('Predict')


class AnalyseForm(FlaskForm):
    submit = SubmitField('Plot')


class AnalyseInitialForm(FlaskForm):
    number = StringField('Number of companies to analyse', validators=[DataRequired()])
    submit = SubmitField('Proceed')


class DistanceForm(FlaskForm):
    dropdown_list = [(1, 'Toyota'), (2, 'General Motors'), (3, 'Apple'), (4, 'Tesla'), (5, 'Google')]
    company1 = SelectField('Company1', choices=dropdown_list, coerce=int, validators=[DataRequired()])
    company2 = SelectField('Company2', choices=dropdown_list, coerce=int, validators=[DataRequired()])
    submit = SubmitField('Calculate')