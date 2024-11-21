import streamlit as st

st.title('🎈Hatem')

st.write('Hello world!')
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# عنوان التطبيق
st.title("تحليل درجات الطلاب في الذكاء الاصطناعي")
st.write("هذا التطبيق يستخدم لتحليل درجات الطلاب وتوقع الأداء بناءً على المدخلات المختلفة.")

# تحميل ملف البيانات
uploaded_file = st.file_uploader("قم بتحميل ملف بيانات الطلاب بصيغة CSV", type="csv")
if uploaded_file is not None:
    # قراءة البيانات
    data = pd.read_csv(uploaded_file)
    st.write("### عرض البيانات الأولية:")
    st.write(data.head())

    # عرض رسم بياني للمجموع الكلي
    st.write("### توزيع الدرجات الكلية:")
    st.bar_chart(data['مجموع'])

    # اختيار الميزات (Features) والهدف (Target)
    features = ['اختبارات قصيرة', 'نشاطات صفية', 'مشاريع', 'واجبات', 'مشاركة وتفاعل']
    target = 'مجموع'

    if st.checkbox("تشغيل تحليل تعلم الآلة"):
        # فصل البيانات إلى تدريب واختبار
        X = data[features]
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # تدريب نموذج الانحدار الخطي
        model = LinearRegression()
        model.fit(X_train, y_train)

        # التوقع باستخدام النموذج
        y_pred = model.predict(X_test)

        # تقييم الأداء
        mse = mean_squared_error(y_test, y_pred)
        st.write("### تقييم النموذج:")
        st.write(f"متوسط الخطأ التربيعي (MSE): {mse:.2f}")

        # توقع درجة طالب جديد
        st.write("### توقع درجة طالب جديد:")
        inputs = {feature: st.number_input(f"أدخل القيمة لـ {feature}:", value=0.0) for feature in features}
        if st.button("توقع"):
            prediction = model.predict([list(inputs.values())])
            st.write(f"الدرجة المتوقعة: {prediction[0]:.2f}")
