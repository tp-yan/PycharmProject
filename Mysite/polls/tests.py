from django.test import TestCase

# 测试系统会自动的在所有以 tests 开头的文件里寻找并执行测试代码
import datetime
from django.utils import timezone
from .models import Question
from django.urls import reverse

# 每次测试进行的时候，django会创建新的数据库，测试完成之后会删除数据库，这样保证每次测试不会污染数据

# 测试Model:需要测试每个方法的正确性
class QuestionModelTests(TestCase):
    def test_was_published_recently_with_future_question(self):
        """
        was_published_recently() returns False for questions whose pub_date
        is in the future.
        """
        time = timezone.now() + datetime.timedelta(days=30)
        future_question = Question(pub_date=time)
        self.assertIs(future_question.was_published_recently(), False)

    def test_was_published_recently_with_old_question(self):
        """
        was_published_recently() returns False for questions whose pub_date
        is older than 1 day.
        """
        time = timezone.now() - datetime.timedelta(days=1, seconds=1)
        old_question = Question(pub_date=time)
        self.assertIs(old_question.was_published_recently(), False)

    def test_was_published_recently_with_recent_question(self):
        """
        was_published_recently() returns True for questions whose pub_date
        is within the last day.
        """
        time = timezone.now() - datetime.timedelta(hours=23,minutes=59, seconds=59)
        recent_question = Question(pub_date=time)
        self.assertIs(recent_question.was_published_recently(), True)



# 创建question对象的快捷函数
def create_question(question_text,days):
    """
        Create a question with the given `question_text` and published the
        given number of `days` offset to now (negative for questions published
        in the past, positive for questions that have yet to be published).
    """
    time = timezone.now() + datetime.timedelta(days=days)
    return Question.objects.create(question_text=question_text,pub_date = time)


# 测试Index视图：主要测试要显示的内容是否与预期相符，以及是否包含了后台传递的数据
class QuestionIndexViewTests(TestCase):
    # 数据库会在每次调用测试方法前被重置(清零)
    def test_no_question(self):
        """
        If no questions exist, an appropriate message is displayed.
        """
        response = self.client.get(reverse('polls:index'))  #  Client 来模拟用户和视图层代码的交互 TestCase类里包含了自己的 client 实例
        # 通过client调用get，post方法获取服务器的返回值response（包含了返还给浏览器的所有内容）
        # response是一个TemplateResponse是经过渲染后的HTML页面以及其他属性，包括HTML页面内容和视图传递给模板的数据等
        self.assertEqual(response.status_code,200)
        self.assertQuerysetEqual(response.context['latest_question_list'],[])
        self.assertContains(response,"No polls are available.")     # response中的所有内容（包括各种属性）中是否包含 目标串

    def test_past_question(self):
        """
        Questions with a pub_date in the past are displayed on the
        index page.
        """
        qp = create_question(question_text="Past question.", days=-30)
        qp.choice_set.create(choice_text="Past", votes=0)
        response = self.client.get(reverse('polls:index'))
        self.assertQuerysetEqual(
            response.context['latest_question_list'],
            ['<Question: Past question.>']
        )

    def test_future_question(self):
        """
        Questions with a pub_date in the future aren't displayed on
        the index page.
        """
        create_question(question_text="Future question.", days=30)
        response = self.client.get(reverse('polls:index'))
        self.assertContains(response, "No polls are available.")
        self.assertQuerysetEqual(response.context['latest_question_list'], [])

    def test_future_question_and_past_question(self):
        """
        Even if both past and future questions exist, only past questions
        are displayed.
        """
        qp = create_question(question_text="Past question.", days=-30)
        qp.choice_set.create(choice_text="Past",votes=0)
        qf = create_question(question_text="Future question.", days=30)
        qf.choice_set.create(choice_text="Future",votes=0)
        response = self.client.get(reverse('polls:index'))
        self.assertQuerysetEqual(
            response.context['latest_question_list'],
            ['<Question: Past question.>']
        )

    def test_two_past_questions(self):
        """
        The questions index page may display multiple questions.
        """
        qp1 = create_question(question_text="Past question 1.", days=-30)
        qp2 = create_question(question_text="Past question 2.", days=-5)
        qp1.choice_set.create(choice_text="Past1", votes=0)
        qp2.choice_set.create(choice_text="Past2", votes=0)
        response = self.client.get(reverse('polls:index'))
        self.assertQuerysetEqual(
            response.context['latest_question_list'],
            ['<Question: Past question 2.>', '<Question: Past question 1.>']
        )

    def test_no_choice_question(self):
        """
        没有选项的问题不应该在Index页显示
        """
        create_question(question_text="No choice.",days=-1)     # 创建一个没有选项的问题
        response = self.client.get(reverse('polls:index'))
        self.assertQuerysetEqual(response.context['latest_question_list'],[])   # 该问题不应该被显示在Index页

# 测试Detail视图
class QuestionDetailViewTests(TestCase):
    def test_future_question(self):
        """
        The detail view of a question with a pub_date in the future
        returns a 404 not found.
        """
        future_question = create_question(question_text='Future question.', days=5)
        # 模拟直接输入URL访问未发表问题：应失败
        url = reverse('polls:detail', args=(future_question.id,))   # /polls/1/
        response = self.client.get(url)
        self.assertEqual(response.status_code, 404)

    def test_past_question(self):
        """
        The detail view of a question with a pub_date in the past
        displays the question's text.
        """
        past_question = create_question(question_text='Past Question.', days=-5)
        url = reverse('polls:detail', args=(past_question.id,))
        response = self.client.get(url)
        self.assertContains(response, past_question.question_text)

class VoteResultsViewTests(TestCase):
    def test_future_question(self):
        """
        不能访问未来问题的投票结果，返回404
        """
        fq = create_question(question_text='Future question.', days=5)
        url = reverse('polls:results',args=(fq.id,))
        response = self.client.get(url)
        self.assertEqual(response.status_code,404)

    def test_past_question(self):
        """
        应该能看到投票结果
        """
        pq = create_question(question_text='Past question.', days=-5)
        ch1 = pq.choice_set.create(choice_text="choice1", votes=1)
        ch2 = pq.choice_set.create(choice_text="choice2", votes=0)
        url = reverse('polls:results', args=(pq.id,))
        response = self.client.get(url)
        self.assertContains(response,ch1.choice_text +" -- "+str(ch1.votes)+" vote")
        self.assertContains(response,ch2.choice_text +" -- "+str(ch2.votes)+" votes")

class VoteTests(TestCase):
    def test_no_question(self):
        """
        测试找不到问题，返回404
        """
        url = reverse("polls:vote",args=(1,))
        response = self.client.post(url,data={'choice':1})
        self.assertEqual(response.status_code,404)


    def test_not_select_choice(self):
        """
        测试没有选择选项，返回error_message
        """
        q = create_question(question_text="question",days=-10)
        q.choice_set.create(choice_text="choice1",votes=0)
        url = reverse('polls:vote',args=(q.id,))
        response = self.client.post(url,data={})
        self.assertEquals(response.context['error_message'],"You didn't select a choice.")

    def test_select_choice(self):
        """
        测试正常投票
        """
        q = create_question(question_text="question", days=-10)
        c1 = q.choice_set.create(choice_text="choice1", votes=0)
        c2= q.choice_set.create(choice_text="choice2", votes=0)

        url = reverse('polls:vote', args=(q.id,))
        response = self.client.post(url, data={"choice":c1.id},follow=True)
        # follow=True：client会追踪任何重定向，返回的response有redirect_chain属性，包括所有重定向过程中的url和状态码组成的元祖列表。
        c1_new = q.choice_set.get(id=c1.id)
        c2_new = q.choice_set.get(id=c2.id)
        self.assertEquals(c1_new.votes,1)
        self.assertEquals(c2_new.votes,0)
        # print(response.redirect_chain[0])
        # print(response.redirect_chain[0][0])
        self.assertEquals(response.redirect_chain[0][0],reverse('polls:results',args=(q.id,)))

