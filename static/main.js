//links
//http://eloquentjavascript.net/09_regexp.html
//https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Regular_Expressions
nlp = window.nlp_compromise;

var messages = [], //array that hold the record of each string in chat
  lastUserMessage = "", //keeps track of the most recent input string from the user
  botMessage = "", //var keeps track of what the chatbot is going to say
  botName = 'QLBot'; //name of the chatbot
//

//edit this function to change what the chatbot says
function chatbotResponse(user_message) {
  botMessage = "I'm confused"; //the default message
  $.getJSON( "http://localhost:5000", {"message": user_message}, function( data ) {
    botMessage = data['message'];
    userMessage = "<p class='triangle-border left'><b>You:</b> " + user_message + "</p>"
    $('#entries').append(userMessage);
    $('#entries').append("<p class='triangle-border right'><b>" + botName + ":</b> " + botMessage + "</p>");
    $("#entries").animate({ scrollTop: $('#entries').prop("scrollHeight")}, 200);
  });
}

//this runs each time enter is pressed.
//It controls the overall input and output
function newEntry() {
  //if the message from the user isn't empty then run
  if (document.getElementById("chatbox").value != "") {
    //pulls the value from the chatbox ands sets it to lastUserMessage
    lastUserMessage = $("#chatbox")[0].value;
    //sets the chat box to be clear
    document.getElementById("chatbox").value = "";
    //adds the value of the chatbox to the array messages
    //sets the variable botMessage in response to lastUserMessage
    chatbotResponse(lastUserMessage);
  }
}



//runs the keypress() function when a key is pressed
document.onkeypress = keyPress;
//if the key pressed is 'enter' runs the function newEntry()
function keyPress(e) {
  var x = e || window.event;
  var key = (x.keyCode || x.which);
  if (key == 13 || key == 3) {
    //runs this function when enter is pressed
    newEntry();
  }
}

//clears the placeholder text ion the chatbox
//this function is set to run when the users brings focus to the chatbox, by clicking on it
function placeHolder() {
  document.getElementById("chatbox").placeholder = "";
}