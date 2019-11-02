from django.urls import path

from . import views

app_name = 'polls'  # 因为在模板中使用{%url%}时，有可能多个 应用中都有name为detail的 URL，故避免重复，需要给URL加上命名空间，即添加这行

urlpatterns = [
    # path(route, view, kwargs=None, name=None)
    # route:一个匹配 URL 的准则（类似正则表达式）,不会匹配 GET 和 POST 参数或域名
    # view:当 Django 找到了一个匹配的准则，就会调用这个特定的视图函数，并传入一个 HttpRequest 对象作为第一个参数，被“捕获”的参数以关键字参数的形式传入。
    # kwargs:任意个关键字参数可以作为一个字典传递给目标视图函数
    # name:为URL 取别名,能使你在 Django 的任意地方唯一地引用它，尤其是在模板中。
    path('', views.IndexView.as_view(), name='index'),
    # /polls/34/:在找到匹配项 'polls/'，它切掉了匹配的文本（"polls/"），将剩余文本——"34/"，发送至 'polls.urls' URLconf 做进一步处理。
    # 在这里剩余文本匹配了 '<int:question_id>/'以关键字参数的形式发送给视图函数， Django 以如下形式调用：
    # detail(request=<HttpRequest object>, question_id=34)
    path('<int:pk>/',views.DetailView.as_view(),name="detail"), # DetailView 期望从 URL 中捕获名为 "pk" 的主键值
    path('<int:pk>/results/',views.ResultsView.as_view(),name="results"),
    path('<int:question_id>/vote/',views.vote,name="vote"),
]