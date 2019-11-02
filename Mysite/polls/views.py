from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse
from django.views import generic    # 导入通用视图模块
from .models import Question,Choice
from django.utils import timezone

'''
def index(request):
    # 展示数据库里以发布日期排序的最近 5 个投票问题
    latest_question_list = Question.objects.order_by('-pub_date')[:5]   # -pub_date:时间逆序排列
    context = {'latest_question_list':latest_question_list}
    return render(request,'polls/index.html',context)

def detail(request, question_id):
    question = get_object_or_404(Question,pk=question_id)  # 类似的有：get_list_or_404(MyModel, published=True)
    return render(request,'polls/detail.html',{'question':question})

def results(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    return render(request, 'polls/results.html', {'question': question})
'''

# 使用通用视图来改良 index detail results
class IndexView(generic.ListView):                  # 通用视图ListView:显示一个对象列表
    template_name = 'polls/index.html'              # 指定视图使用模板，覆盖默认生成的模板名
    context_object_name = 'latest_question_list'    # 传递给模板字典数据的键名

    # ListView子类必须重写get_queryset方法
    def get_queryset(self):
        """返回最新五个问题"""             # 筛选出小于等于当前日期的数据
        # qs = Question.objects.filter(pub_date__lte=timezone.now()).order_by('-pub_date')[:5]
        queryset = Question.objects.filter(pub_date__lte=timezone.now()).order_by('-pub_date')[:5]
        has_choices = []
        for entry in queryset:
            if entry.choice_set.count() > 0:
                has_choices.append(entry)
        return has_choices

class DetailView(generic.DetailView):   # DetailView:显示一个特定类型对象的详细信息页面
    model = Question                    # 通用视图需要知道它将作用于哪个模型。 这由 model 属性提供
    template_name = "polls/detail.html"
    # 在通用视图内可以使用 self.request
    # DetailView：将会在get_queryset的返回结果中再去查找指定pk的对象(get_object_or_404)，如果没有找到则返回404
    def get_queryset(self):
        """
        Excludes any questions that aren't published yet.
        """
        return Question.objects.filter(pub_date__lte=timezone.now())


class ResultsView(generic.DetailView):
    model = Question                    # 同时也决定了传给模板的context变量中，键为‘question’
    template_name = "polls/results.html"

    def get_queryset(self):
        """
        Excludes any questions that aren't published yet.
        """

        return Question.objects.filter(pub_date__lte=timezone.now())


def vote(request, question_id):
    question = get_object_or_404(Question,pk=question_id)
    try:
        selected_choice = question.choice_set.get(pk=request.POST['choice'])    # request.POST 是一个类字典对象, request.POST 的值永远是字符串
    except (KeyError,Choice.DoesNotExist):
        return render(request,'polls/detail.html',{
            'question':question,
            'error_message':"You didn't select a choice."
        })
    else:
        selected_choice.votes += 1
        selected_choice.save()
        # 处理完POST请求后一定要重定向，防止用户返回页面再次提交
        # reverse() 函数避免了我们在视图函数中硬编码 URL，需要给出想要跳转的视图的名字和该视图所对应的URL模式中需要给该视图提供的参数
        # reverse() 调用将返回一个这样的字符串：'/polls/3/results/'
        return HttpResponseRedirect(reverse('polls:results',args=(question.id,)))

