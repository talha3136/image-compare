from rest_framework import viewsets, mixins, status
from rest_framework.permissions import IsAuthenticated, AllowAny, IsAdminUser
from .models import User
from .serializers import ChangeSelfPasswordSerializer, ChangeTeamMemberPasswordSerializer, LoginSerializer, SuperUserSerializer, UserSerializer, logoutSerializer
from rest_framework.response import Response 
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.decorators import action
from home.services.auth_services import AuthenticationService
from rest_framework_simplejwt.tokens import RefreshToken, AccessToken
from django.shortcuts import get_object_or_404



class AuthViewSet(viewsets.GenericViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = (AllowAny)

    @action(
        detail=False,
        methods=('POST',),
        url_path='login',
        serializer_class=LoginSerializer,
    )
    def login(self, request):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        email, password = (
            serializer.validated_data['email'],
            serializer.validated_data['password'],
        )

        access_token, refresh_token, user = AuthenticationService.login(email, password, is_superuser=False)

        # LoginHistoryService.log_user_login(
        #     user=user,
        #     x_forwarded_for=request.META.get('HTTP_X_FORWARDED_FOR'),
        #     x_remote_addr=request.META.get('REMOTE_ADDR'),
        #     user_agent_info=request.META.get('HTTP_USER_AGENT', '<unknown>'),
        # )

        return Response(
            data={
                'access': access_token,
                'refresh': refresh_token,
                'user': UserSerializer(user).data
            },
            status=status.HTTP_200_OK,
        )
    
    @action(
        detail=False,
        methods=('POST',),
        url_path='login-superuser',
        serializer_class=LoginSerializer,
    )
    def login_superuser(self, request):

        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        email, password = (
            serializer.validated_data['email'],
            serializer.validated_data['password'],
        )
 
        # if not AuthenticationService.verify_recaptcha(serializer.validated_data['recaptchaToke']):
        #     return Response(data="Invalid recapcha toke", status=status.HTTP_400_BAD_REQUEST)

        access_token, refresh_token, user = AuthenticationService.login(email, password, is_superuser=True)
        
        return Response(
            data={
                'access': access_token,
                'refresh': refresh_token,
                'user': UserSerializer(user).data,
            },
            status=status.HTTP_200_OK,
        )
    @action(
        detail=False,
        methods=('POST',),
        url_path='logout',
        serializer_class=logoutSerializer,
        permission_classes=([IsAuthenticated])

    )
    def logout_user(self, request):
        """Blacklist the refresh token: extract token from the header
        during logout request user and refresh token is provided"""
        Refresh_token = request.data["refresh"]
        token = RefreshToken(Refresh_token)
        token.blacklist()
     
        return Response("Successful Logout", status=status.HTTP_200_OK)
    
    @action(detail=False, 
        methods=('POST',), 
        url_path=r'change-team-member-password', 
        serializer_class=ChangeTeamMemberPasswordSerializer, 
        permission_classes=([IsAuthenticated])
    )
    def change_team_member_password(self, request):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        password, user  = (
            serializer.validated_data['password'],
            serializer.validated_data['user'],
        )
        arezUser = get_object_or_404(User, pk=user)
        arezUser.set_password(password)
        arezUser.save()
        return Response("password change successfully" ,status=status.HTTP_200_OK)
    

    @action(
        detail=False, 
        methods=('POST',), 
        url_path=r'change-self-password', 
        serializer_class=ChangeSelfPasswordSerializer, 
        permission_classes=([IsAuthenticated])
    )
    def change_self_password(self, request):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        password, user  = (
            serializer.validated_data['password'],
            serializer.validated_data['user'],
        )
        arezUser = get_object_or_404(User, pk=user)
        arezUser.set_password(password)
        arezUser.save()
        return Response("password change successfully" ,status=status.HTTP_200_OK)

class UserTokenViewSet(viewsets.GenericViewSet, mixins.ListModelMixin):
    serializer_class = UserSerializer
    permission_classes = [IsAuthenticated]
    def get_queryset(self):
        email = self.request.user.email
        queryset = User.objects.filter(email=email)
        return queryset


class UserViewSet(viewsets.GenericViewSet, mixins.RetrieveModelMixin, mixins.UpdateModelMixin, mixins.DestroyModelMixin):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = ([IsAuthenticated ])
    http_method_names = ['get','post','put', 'delete','patch']

    def create(self, request, *args, **kwargs):
        data = request.data.copy()
        data['is_active'] = True 

        serializer = self.get_serializer(data=data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save() 
        user_serializer = self.get_serializer(user)
        serialized_user = user_serializer.data
        return Response(serialized_user, status=status.HTTP_201_CREATED)
    
    @action(
        detail=False, 
        methods=['post'], 
        url_path= 'create-superuser',
        permission_classes=[IsAuthenticated],
        serializer_class = SuperUserSerializer,
        parser_classes = (FormParser, MultiPartParser)
        )
    def create_superuser(self, request):
        data = request.data.copy()
        data['is_superuser'] = True
        data['is_staff'] = True
        data['is_active'] = True

        serializer = UserSerializer(data=data)
        serializer.is_valid(raise_exception=True)
        superuser = serializer.save()

        return Response(self.get_serializer(superuser).data, status=status.HTTP_201_CREATED)