$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();
    $('#image-container').hide();
    $('#btn-predict').hide()
    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }

    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#image-container').hide();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });
    function checkImg(id){
        var oldImg = document.getElementById(id).querySelector('img');
         if (oldImg != null) {
            oldImg.parentNode.removeChild(oldImg);
         }
    }

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);
        // Show loading animation
        $(this).hide();
        $('.loader').show();
        $('.image-section').hide();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#result').fadeIn(600);
                console.log('Received response:', data);
                var num = Object.keys(data).length;
                if(num == 2){
                    $('#image-container').show()
                    $('#result').text(' Result:  ' + data.result);
                     var imgData = 'data:image/png;base64,' + data.image;
                     var img = document.createElement('img');
                     img.src = imgData;
                     checkImg('image-container')
                     document.getElementById('image-container').appendChild(img);
                    console.log('Success!');
                }
                else {
                    checkImg('image-container')
                    $('#result').text(' Result:  ' + data);
                }
            },
        });
    });
});
