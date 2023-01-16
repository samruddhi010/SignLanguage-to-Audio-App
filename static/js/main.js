$(document).ready(function(){
    console.log("start of everything")
    var socket = io('http://localhost:5000', {transports: ['websocket', 'polling', 'flashsocket']})

    socket.on('connect', function(){
        console.log("Connected...!", socket.connected)
    });

    const video = document.querySelector("#videoElement");

    video.width = 500; 
    video.height = 375;

    if (navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
        .then(function (stream) {
            video.srcObject = stream;
            video.play();
        })
        .catch(function (err0r) {
            console.log(err0r)
            console.log("Something went wrong!");
        });
    }

    let src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
    let dst = new cv.Mat(video.height, video.width, cv.CV_8UC1);
    let cap = new cv.VideoCapture(video);

    const FPS = 22;

    setInterval(() => {
        cap.read(src);

        var type = "image/png"
        var data = document.getElementById("videoArea").toDataURL(type);
        data = data.replace('data:' + type + ';base64,', ''); //split off junk at the beginning

        socket.emit('image', data);
    }, 10000/FPS);


    socket.on('response_back', function(image){
        const image_id = document.getElementById('image');
        image_id.src = image;
    });

    socket.on('update_sign', function(sign){
        changeSign(sign);
    });
});


var SignsList = Array("No", "Yes", "Thankyou", "Goodbye", "Hello", "Iloveyou", "Please", "Sorry", "Youarewelcome");

function changeSign(value) 
{
    activeSign = value


    console.log(activeSign)


    if (activeSign == "No") {
        document.getElementById("signSelected").src="{{url_for('static', filename='images/no-signlanguage.png')}}";
        document.getElementById('signText').innerHTML = 'No';
    }
    if (activeSign == "Yes") {
        document.getElementById("signSelected").src="{{url_for('static', filename='images/yes-signlanguage.png')}}"
        document.getElementById('signText').innerHTML = 'Yes';
    }
    if (activeSign == "Thankyou") {
        document.getElementById("signSelected").src="{{url_for('static', filename='images/thankyou-signlanguage.png')}}"
        document.getElementById('signText').innerHTML = 'Thank you';
    }
    if (activeSign == "Goodbye") {
        document.getElementById("signSelected").src="{{url_for('static', filename='images/goodbye-signlanguage.png')}}"
        document.getElementById('signText').innerHTML = 'Goodbye';
    }
    if (activeSign == "Hello") {
        document.getElementById("signSelected").src="{{url_for('static', filename='images/hello-signlanguage.png')}}"
        document.getElementById('signText').innerHTML = 'Hello';
    }
    if (activeSign == "Iloveyou") {
        document.getElementById("signSelected").src="{{url_for('static', filename='images/iloveyou-signlanguage.png')}}"
        document.getElementById('signText').innerHTML = 'I love you';
    }
    if (activeSign == "Please") {
        document.getElementById("signSelected").src="{{url_for('static', filename='images/please-signlanguage.png')}}"
        document.getElementById('signText').innerHTML = 'Please';
    }
    if (activeSign == "Sorry") {
        document.getElementById("signSelected").src="{{url_for('static', filename='images/sorry-signlanguage.png')}}"
        document.getElementById('signText').innerHTML = 'Sorry';
    }
    if (activeSign == "Youarewelcome") {
        document.getElementById("signSelected").src="{{url_for('static', filename='images/youarewelcome-signlanguage.png')}}"
        document.getElementById('signText').innerHTML = 'You are welcome';
    }
}

function playSound() 
{
    if (activeSign == "No") {
        document.getElementById("NoSound").play();
    }
    if (activeSign == "Yes") {
        document.getElementById("YesSound").play();
    }
    if (activeSign == "Thankyou") {
        document.getElementById("ThankyouSound").play();
    }
    if (activeSign == "Goodbye") {
        document.getElementById("GoodbyeSound").play();
    }
    if (activeSign == "Hello") {
        document.getElementById("HelloSound").play();
    }
    if (activeSign == "Iloveyou") {
        document.getElementById("IloveyouSound").play();
    }
    if (activeSign == "Please") {
        document.getElementById("PleaseSound").play();
    }
    if (activeSign == "Sorry") {
        document.getElementById("SorrySound").play();
    }
    if (activeSign == "Youarewelcome") {
        document.getElementById("YouarewelcomeSound").play();
    }
}